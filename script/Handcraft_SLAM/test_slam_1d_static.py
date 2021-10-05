import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import os
from slam_1d_no_map import RobotMove
import statistics
sys.path.append('../../Env/1D/')
# sys.path.append('1D/Static/')
from DMP_Env_1D_static_test import deep_mobile_printing_1d1r

save_path = "./log/SLAM/plot/1D/Static/with_out_map/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)

print('1D_Static')

iou_all_average = 0
iou_all_min = 1
std_his=[]
for plan_choose in range(3):
    iou_his=[]

    env = deep_mobile_printing_1d1r(plan_choose=plan_choose)
    ######################
    # hyper parameter
    minibatch_size = 64
    Lr = 0.00001
    N_iteration = 6000
    N_iteration_test = 500
    alpha = 0.9
    Replay_memory_size = 1000
    Update_traget_period = 500
    Action_dim = env.action_dim
    State_dim = env.state_dim
    hidden_state_dim = 256
    Time_step = 20
    UPDATE_FREQ = 5
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    ######################
    PLAN_DIM = env.plan_width
    print("State_dimension:", State_dim)
    print("Total Brick: ", env.total_brick)
    print("plan_width", PLAN_DIM)
    print('######################')

    
    best_iou = 0
    iou_test_total = 0
    iou_min = 1
    reward_test_total = 0
    best_env = np.array([])
    start_time_test = time.time()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    

    for ep in range(N_iteration_test):
        print(ep)
        state = env.reset()
        reward_test = 0
        robot_decider = RobotMove(env)

        while True:
            
            robot_decider.step()
            state_next, r, done = robot_decider.observation, robot_decider.reward, robot_decider.done

            reward_test += r
            if done:
                break
            state = state_next

        iou_test = env.iou()
        iou_his.append(iou_test)
        iou_min = min(iou_min, iou_test)

        if iou_test > best_iou:
            best_iou = iou_test
            best_plan = env.plan
            best_step = env.count_step
            best_brick = env.count_brick
            best_tb = env.total_brick
            best_env = env.environment_memory
        iou_test_total += iou_test
        reward_test_total += reward_test

    reward_test_total = reward_test_total / N_iteration_test
    iou_test_total = iou_test_total / N_iteration_test
    secs = int(time.time() - start_time_test)
    mins = secs / 60
    secs = secs % 60
    std = statistics.pstdev(iou_his)
    std_his.append(std)
    # env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)
    iou_all_average += iou_test_total
    iou_all_min = min(iou_min,iou_all_min)
    # plt.savefig(save_path+"Plan"+str(plan_choose)+'.png')

iou_all_average = iou_all_average/3
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
print("std",std_his)
# plt.show()


#with map
# iou_all_average 0.9456223242900906
# iou_all_min 0.4278523489932886

#no map
# iou_all_average 0.6764667976751952
# iou_all_min 0.4457831325301205
