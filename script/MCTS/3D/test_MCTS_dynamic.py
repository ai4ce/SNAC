import os
import sys
import torch
import torch.nn as nn
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from collections import deque
import sys
pth_plan = ['18', '299']
sys.path.append('../../../Env/3D/')
sys.path.append('../utils')
from DMP_simulator_3d_dynamic_triangle_MCTS_test import deep_mobile_printing_3d1r
import uct_dynamic_inputplan
save_path = "./plot/Dynamic/"
load_path = "./log/dynamic/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('3D_Dynamic')

iou_all_average = 0
iou_all_min = 1

for test_set in range(10):
    plan_choose=1 ## 0:densetriangle 1:sparsetriangle
    env = deep_mobile_printing_3d1r(plan_choose=plan_choose, test_set=test_set)
    ######################
    # hyper parameter
    minibatch_size = 2000
    Lr = 0.0001
    N_iteration = 3000
    N_iteration_test = 200
    ROLLOUT=20
    UCB_CONSTANT=0.5
    alpha = 0.9
    Replay_memory_size = 50000
    Update_traget_period = 200
    UPDATE_FREQ = 1
    Action_dim = env.action_dim
    State_dim = env.state_dim
    plan_dim = env.plan_width
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    ######################
    print("State_dim:",State_dim)
    print("total_step:",env.total_step)
    print("Action_dim",Action_dim)

    def iou(environment_memory,environment_plan,HALF_WINDOW_SIZE,plan_height,plan_width):
        component1=environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                           HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
        component2=environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                           HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
        overlap = component1*component2 # Logical AND
        union = component1 + component2 # Logical OR
        IOU = overlap.sum()/float(union.sum())
        return IOU

    def get_and_init_FC_layer(din, dout):
        li = nn.Linear(din, dout)
        nn.init.xavier_uniform_(
           li.weight.data, gain=nn.init.calculate_gain('relu'))
        li.bias.data.fill_(0.)
        return li

    class Q_NET(nn.Module):
        def __init__(self, ):
            super(Q_NET, self).__init__()
            self.conv_layer1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=2) # potential check - in_channels
            self.conv_layer2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2)
            self.conv_layer3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2)
            self.conv_layer4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2)
            self.fc_1 = get_and_init_FC_layer(State_dim+1+32, 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.out = get_and_init_FC_layer(128, 1)
            self.relu = nn.ReLU()
        def forward(self, s,a,plan,bsize):
            plan = plan.view(bsize,1,20,20) 
            conv_out = self.conv_layer1(plan)
            conv_out = self.relu(conv_out)
            conv_out = self.conv_layer2(conv_out)
            conv_out = self.relu(conv_out)
            conv_out = self.conv_layer3(conv_out)
            conv_out = self.relu(conv_out)
            conv_out = conv_out.view(bsize,32)
            ## concate with obs
            x=torch.cat((s,a,conv_out),dim=1)
            x = self.fc_1(x)
            x = self.relu(x)
            x = self.fc_2(x)
            x = self.relu(x)
            x = self.fc_3(x)
            x = self.relu(x)
            Q = self.out(x)
            return Q
    UCT_mcts = uct_dynamic_inputplan.UCT(
    action_space=env.action_space,
    rollouts=ROLLOUT,
    horizon=100,
    ucb_constant=UCB_CONSTANT,
    is_model_dynamic=True
    )

    class DQN_AGNET():
        def __init__(self,device):
            self.Eval_net= Q_NET().to(device)
            self.Target_net = Q_NET().to(device)
            self.learn_step = 0                                     # counting the number of learning for update traget periodiclly 
            self.count_memory = 0                                         # counting the transitions 
            self.replay_memory = deque(maxlen=Replay_memory_size)    
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
            self.greedy_epsilon=0.2
            self.loss = nn.SmoothL1Loss()
            self.loss_his=[]
            self.device=device
        def store_memory(self,s,a,r,s_next,env_plan):
            self.replay_memory.append((s,a,r,s_next,env_plan))
            self.count_memory+=1
        def choose_action(self,env,state,obs,done):
            action=UCT_mcts.act(env,self.Eval_net,state,obs,done,self.device)   
            return action
        def learning_process(self):
            self.optimizer.zero_grad()
            self.Eval_net.train()
            if self.learn_step% Update_traget_period == 0:
                self.Target_net.load_state_dict(self.Eval_net.state_dict())
            minibatch = random.sample(self.replay_memory,minibatch_size)
            current_states = []
            acts = []
            rewards = []
            next_states = []
            env_plans = []
            for b in minibatch:
                current_states.append(b[0])
                acts.append(b[1])
                rewards.append(b[2])
                next_states.append(b[3])
                env_plans.append(b[4])
            current_states = np.array(current_states)
            # print("current_states_shape",current_states.shape)
            acts = np.array(acts).reshape(minibatch_size,1)
            # print("action_shape",acts.shape)
            rewards = np.array(rewards).reshape(minibatch_size,1)
            next_states = np.array(next_states)
            env_plans = np.array(env_plans)
            b_state = torch.from_numpy(current_states).float().to(device).squeeze(1)
            b_action = torch.from_numpy(acts).float().to(device)
            b_reward = torch.from_numpy(rewards).float().to(device)
            b_state_next = torch.from_numpy(next_states).float().to(device).squeeze(1)
            b_env_plans = torch.from_numpy(env_plans).float().to(device)
            Q_eval = self.Eval_net(b_state,b_action,b_env_plans,minibatch_size)
            Q=torch.zeros(minibatch_size,Action_dim).to(device)
            for item in range(Action_dim):
                a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                a=a.expand(minibatch_size,-1)
                Q_value=self.Target_net(b_state_next,a,b_env_plans,minibatch_size).detach().squeeze(1)
                Q[:,item]=Q_value 
            Q_next=torch.max(Q,dim=1)[0].reshape(minibatch_size,1)
            Q_target = b_reward + alpha * Q_next
            loss = self.loss(Q_eval, Q_target)
            loss.backward()
            self.optimizer.step()
            self.learn_step+=1       
            train_loss=loss.item()
            return train_loss

    PLAN_LIST=["densetriangle","sparsetriangle"]
    PLAN_NAME=PLAN_LIST[plan_choose]
    OUT_FILE_NAME="DQN_3d_"+PLAN_NAME+"_lr"+str(Lr)+"_rollouts"+str(ROLLOUT)+"_ucb_constant"+str(UCB_CONSTANT)
    test_agent=DQN_AGNET(device)
    log_path = load_path + OUT_FILE_NAME + "/" + "Eval_net_episode_"
    test_agent.Eval_net.load_state_dict(torch.load(log_path + pth_plan[plan_choose]+".pth" ,map_location=device))
    best_iou=0
    best_env = np.array([])
    iou_test_total=0
    reward_test_total=0
    iou_min = 1
    iou_history = []
    start_time_test = time.time()
    fig = plt.figure(figsize=[10, 5])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(N_iteration_test):
        state, obs = env.reset()
        reward_test=0
        print('Ep:',i)
        done=False
        while True:            
            action = test_agent.choose_action(env,state,obs,done)
            state_next,obs_next, r, done = env.step(action)
            reward_test += r
            if done:
                break
            state = state_next
            obs = obs_next 
        iou_test=env.iou()
        iou_history.append(iou_test)
        iou_min = min(iou_min,iou_test)
        if iou_test > best_iou:
            best_iou = iou_test
            best_step = env.count_step
            best_brick = env.count_brick
            best_plan = env.plan
            best_tb = env.total_brick
            best_env = env.environment_memory
            best_posi = env.position_memory[-1]
        iou_test_total+=iou_test
        reward_test_total+=reward_test
    reward_test_total=reward_test_total/N_iteration_test
    iou_test_total=iou_test_total/N_iteration_test
    secs = int(time.time() - start_time_test)
    mins = secs / 60
    secs = secs % 60
    iou_all_average += iou_test_total
    iou_all_min = min(iou_min,iou_all_min)
    print("total_brick:", env.total_brick)
    print('iou_his:',iou_history)
    print('iou_min:',min(iou_history))
    if best_iou>0:
        env.render(ax1,ax2,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,
               best_iou=best_iou,best_step=best_step,best_brick=best_brick,position_memo = best_posi)
    else:
        env.render(ax1,ax2,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)
    plt.savefig(save_path + "Plan" + str(plan_choose) +'_test'+str(test_set) + '.png')

iou_all_average = iou_all_average/20
print('#### Finish #####')
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()