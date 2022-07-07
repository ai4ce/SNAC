import gym
import torch
from gym import spaces
from collections import defaultdict
import heapq
import pickle
import matplotlib.pyplot as plt
import joblib
import numpy as np
import sys
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_dynamic import deep_mobile_printing_2d1r
import heapq 
import os

def localization(action, obs, new_obs, previous_position):
    [y,x]=previous_position
    # 0 left, 1 right, 2 up, 3 down, 4 drop
    obs = obs[0][:49].reshape(7, 7)
    new_obs = new_obs[0][:49].reshape(7, 7)
    size_x = len(obs)
    size_y = len(obs[0])

    new_x, new_y = x, y
    dx = [-1, 1, 0, 0]
    dy = [0, 0, 1, -1]
    if action == 4:
        return new_y, new_x
    elif (obs == new_obs).all():
        #print('enter action_none')
        new_x = x + dx[action]
        new_y = y + dy[action]


    else:
        if action == 0:
            if (obs[:, :size_x + dx[action]] == new_obs[:, -dx[action]:]).all():
                #print('enter action0_1')
                new_x = x + dx[action]
            elif (obs[:, :size_x + 2 * dx[action]] == new_obs[:, -2 * dx[action]:]).all():
                #print('enter action0_2')
                new_x = x + 2 * dx[action]
            elif (obs[:, :size_x + 3 * dx[action]] == new_obs[:, -3 * dx[action]:]).all():
                #print('enter action0_3')
                new_x = x + 3 * dx[action]
        elif action == 1:  # right
            if (new_obs[:, :size_x - dx[action]] == obs[:, dx[action]:]).all():
                #print('enter action1_1')
                new_x = x + dx[action]
            elif (new_obs[:, :size_x - 2 * dx[action]] == obs[:, 2 * dx[action]:]).all():
                #print('enter action1_2')
                new_x = x + 2 * dx[action]
            elif (new_obs[:, :size_x - 3 * dx[action]] == obs[:, 3 * dx[action]:]).all():
                #print('enter action1_3')
                new_x = x + 3 * dx[action]
        elif action == 2:  # up
            if (new_obs[:size_y - dy[action], :] == obs[dy[action]:, :]).all():
                #print('enter action2_1')
                new_y = y + dy[action]
            elif (new_obs[:size_y - 2 * dy[action], :] == obs[2 * dy[action]:, :]).all():
                #print('enter action2_2')
                new_y = y + 2 * dy[action]
            elif (new_obs[:size_y - 3 * dy[action], :] == obs[3 * dy[action]:, :]).all():
                #print('enter action2_3')
                new_y = y + 3 * dy[action]
        elif action == 3:
            if (obs[:size_y + dy[action], :] == new_obs[-dy[action]:, :]).all():
                #print('enter action3_1')
                new_y = y + dy[action]
            elif (obs[:size_y + 2 * dy[action], :] == new_obs[-2 * dy[action]:, :]).all():
                #print('enter action3_2')
                new_y = y + 2 * dy[action]
            elif (obs[:size_y + 3 * dy[action], :] == new_obs[-3 * dy[action]:, :]).all():
                #print('enter action3_3')
                new_y = y + 3 * dy[action]

    if new_x < 0: new_x = 0
    if new_x >= env.plan_width: new_x = env.plan_width - 1
    if new_y < 0: new_y = 0
    if new_y >= env.plan_width: new_y = env.plan_width - 1

    # if the robot can see the wall, it can localize its position
    dff_y = 0
    dff_x = 0
    mid = size_x // 2

    if new_obs[0][mid] == -1.0:  # check upper_all
        #print('check_up')
        for i in range(1, env.HALF_WINDOW_SIZE):
            if new_obs[i][mid] == -1.0:
                dff_y -= 1
        new_y = env.HALF_WINDOW_SIZE - 1 + dff_y
    elif new_obs[size_y - 1][mid] == -1.0:  # check down_wall
        #print('check_down')
        for i in range(size_y - 2, env.HALF_WINDOW_SIZE, -1):
            if new_obs[i][mid] == -1.0:
                dff_y += 1
        new_y = env.plan_height - env.HALF_WINDOW_SIZE + dff_y
    if new_obs[mid][0] == -1.0:  # check left_wall
        #print('check_left')
        for i in range(1, env.HALF_WINDOW_SIZE):
            if new_obs[mid][i] == -1.0:
                dff_x -= 1
        new_x = env.HALF_WINDOW_SIZE - 1 + dff_x
    elif new_obs[mid][size_x - 1] == -1.0:  # check right_wall
        #print('check_right')
        for i in range(size_x - 2, env.HALF_WINDOW_SIZE, -1):
            if new_obs[mid][i] == -1.0:
                dff_x += 1
        new_x = env.plan_width - env.HALF_WINDOW_SIZE + dff_x
    
    return [new_y, new_x]


def planing(position,obs,current_action_prior):
    ###   reshape the input value 
    obs = obs[0][:49].reshape(7, 7)
    current_position_row = position[0] + env.HALF_WINDOW_SIZE
    current_position_column = position[1] + env.HALF_WINDOW_SIZE
    
    ######### determine the current action prior 
    
    if (obs[:,0] == -1).all() or (obs[:,0:1] == -1).all()  or (obs[:,0:2] == -1).all(): ## in this case, robot observe left wall, so move right will be advanced
        current_action_prior[0]=0
        current_action_prior[1]=0.5
    if (obs[0,:] == -1).all() or (obs[0:1,:] == -1).all()  or (obs[0:2,:] == -1).all(): ## in this case, robot observe bottom wall, so move down will be advanced
        current_action_prior[2]=0.5
        current_action_prior[3]=0
    if (obs[:,6] == -1).all() or (obs[:,5:6] == -1).all()  or (obs[:,4:6] == -1).all(): ## in this case, robot observe right wall, so move left will be advanced
        current_action_prior[0]=0.5
        current_action_prior[1]=0
    if (obs[6,:] == -1).all() or (obs[5:6,:] == -1).all()  or (obs[4:6,:] == -1).all(): ## in this case, robot observe top wall, so move up will be advanced
        current_action_prior[2]=0
        current_action_prior[3]=0.5
        
    ##### check the observation with the design D 
    distance_array=[]
    for i in range(7):
        for j in range(7):
            if obs[i,j]==-1:
                heapq.heappush(distance_array, (np.inf, [i,j]))
            else:
                if obs[i,j]==0 and env.plan[current_position_row-(3-i),current_position_column-(3-j)]==0:
                    heapq.heappush(distance_array, (np.inf, [i,j]))
                if obs[i,j]==1 and env.plan[current_position_row-(3-i),current_position_column-(3-j)]==0:
                    heapq.heappush(distance_array, (np.inf, [i,j]))
                if obs[i,j]==1 and env.plan[current_position_row-(3-i),current_position_column-(3-j)]==1:
                    heapq.heappush(distance_array, (np.inf, [i,j]))
                if obs[i,j]==0 and env.plan[current_position_row-(3-i),current_position_column-(3-j)]==1:
                    heapq.heappush(distance_array, (np.abs(3-i)+np.abs(3-j), [i,j]))
    
    smallest_distance_candidate=[]
    smallest_distance_candidate.append(heapq.heappop(distance_array))
    if smallest_distance_candidate[0][0]==np.inf:  ### in this case, all the position should not be built and also dont know where to move 
        action = np.random.choice(4, p=current_action_prior) ### so we choose action base on priority 
        return action, current_action_prior  
    else:           ### in this case, we store all all the possible position for next step
        for i in range(49):
            smallest_distance_candidate.append(heapq.heappop(distance_array))
            if smallest_distance_candidate[0][0] != smallest_distance_candidate[-1][0]:
                break
        smallest_distance_candidate.remove(smallest_distance_candidate[-1])
       
    ### in this case we need to decide action base on candidate position 
    if smallest_distance_candidate[0][0]==0:
        action=4 #### in this case, the robot should build brick on its current position 
        return action,current_action_prior 
    else: 
        index=np.random.choice(len(smallest_distance_candidate))
        decided_position=smallest_distance_candidate[index][1]
        
        ##############
    if decided_position[0]>3 and decided_position[1]>3:
        action = np.random.choice([1,2])
        return action,current_action_prior 
    if decided_position[0]>3 and decided_position[1]<3:
        action = np.random.choice([0,2])
        return action,current_action_prior 
    if decided_position[0]<3 and decided_position[1]<3:
        action = np.random.choice([0,3])
        return action,current_action_prior 
    if decided_position[0]<3 and decided_position[1]>3:
        action = np.random.choice([1,3])
        return action,current_action_prior 
    ################ 
    if decided_position[0]==3:
        if decided_position[1]>3:
            action = 1 
            return action,current_action_prior
        else:
            action = 0 
            return action,current_action_prior    
    if decided_position[1]==3:
        if decided_position[0]>3:
            action = 2
            return action,current_action_prior 
        else:
            action = 3 
            return action,current_action_prior  

    
def iou(environment_memory,environment_plan,HALF_WINDOW_SIZE,plan_height,plan_width):
    component1=environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                        HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
    component2=environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                        HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
    overlap = component1*component2 # Logical AND
    union = component1 + component2 # Logical OR
    IOU = overlap.sum()/float(union.sum())
    return IOU

    
    
replay_memory = []
env = deep_mobile_printing_2d1r(plan_choose=0)
iou_test_total = 0
for episode in range(10000):
    print(episode)
    total_reward = 0
    state = env.reset()
    obs=state[0]
    env_plan=state[1]
    current_position=[0,0]
    current_action_prior=[0, 0, 0, 0]
    local_memory = []
    while True:
        action,next_action_prior = planing(current_position,obs,current_action_prior)
        new_state, reward, done = env.step(action)
        new_obs=new_state[0]
        next_position = localization(action, obs, new_obs, current_position)
        local_memory.append((obs, action, reward, new_obs,env_plan))
        total_reward += reward
        obs = new_obs
        current_position=next_position
        current_action_prior=next_action_prior
        if done:
            iou_test=iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
            # print("iou_test",iou_test)
            break
    # print("reward:", total_reward)
    iou_test_total += iou_test
    replay_memory.append(local_memory)
print("iou_test_total",iou_test_total/10000)
save_path = "/mnt/NAS/home/WenyuHan/SNAC/DRQN_expertise/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
joblib.dump(replay_memory, save_path+"replay_memory_2d_dynamic_dense_2.pkl")

# input_memory = []
# for i in range(1, 2):
#     filename = '/mnt/NAS/home/WenyuHan/SNAC/DRQN_expertise/replay_memory_2d_dynamic_dense_%s.pkl' % str(i)
#     local_memories = joblib.load(filename)
#     for local_memory in local_memories:
#         input_memory.append(local_memory)
        
# print(len(input_memory))
# print(len(input_memory[0]))
# print(len(input_memory[1]))