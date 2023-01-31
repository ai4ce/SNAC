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
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_static_test import deep_mobile_printing_2d1r
pth_plan = ['0', '0']
save_path = "./log/plot/2D_Static/"
load_path = "./log/2D_Static/plan_"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('2D_Static')
iou_all_average = 0
iou_all_min = 1

for plan_choose in range(2):
    env = deep_mobile_printing_2d1r(plan_choose=plan_choose)
    ######################
    # hyper parameter
    minibatch_size = 2000
    Lr = 0.001
    N_iteration = 3000
    N_iteration_test = 500
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
        li.weight.data.normal_(0, 0.1)
        return li

    class Q_NET(nn.Module):
        def __init__(self, ):
            super(Q_NET, self).__init__()
            self.fc_1 = get_and_init_FC_layer(State_dim + 1 , 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.out = get_and_init_FC_layer(128, 1)
            self.relu = nn.ReLU()
        def forward(self, s, a):
            x = torch.cat((s, a), dim=1)
            x = self.fc_1(x)
            x = self.relu(x)
            x = self.fc_2(x)
            x = self.relu(x)
            x = self.fc_3(x)
            x = self.relu(x)
            Q = self.out(x)
            return Q

    class DQN_AGNET():
        def __init__(self, device):
            self.Eval_net = Q_NET().to(device)
            self.Target_net = Q_NET().to(device)
            self.learn_step = 0  # counting the number of learning for update traget periodiclly
            self.count_memory = 0  # counting the transitions
            self.replay_memory = deque(maxlen=Replay_memory_size)
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
            self.greedy_epsilon = 0.2
            self.loss = nn.SmoothL1Loss()
            self.loss_his = []
        def store_memory(self, s, a, r, s_next, env_plan):
            self.replay_memory.append((s, a, r, s_next))
            self.count_memory += 1
        def choose_action(self, s):
            state = torch.FloatTensor(s).unsqueeze(0).to(device)
            Q = []
            for item in range(Action_dim):
                a = torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                Q_value = self.Eval_net(state, a)
                Q.append(Q_value.item())
            action = np.argmax(Q)
            return action
    test_agent=DQN_AGNET(device)
    log_path = load_path + str(plan_choose) + "/" + "Eval_net_episode_"
    test_agent.Eval_net.load_state_dict(torch.load(log_path + pth_plan[plan_choose]+".pth" ,map_location='cuda:0'))
    best_iou=0
    best_env = np.array([])
    iou_test_total=0
    reward_test_total=0
    iou_min = 1
    iou_history = []
    start_time_test = time.time()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(N_iteration_test):
        state = env.reset()
        reward_test=0
        print('Ep:',i)
        while True:
            state = state.reshape(state.shape[-1])
            action = test_agent.choose_action(state)
            state_next, r, done = env.step(action)
            state_next = state_next.reshape(state_next.shape[-1])
            reward_test += r
            if done:
                break
            state = state_next
        iou_test=iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
        iou_history.append(iou_test)
        iou_min = min(iou_min,iou_test)
        if iou_test > best_iou:
            best_iou = iou_test
            best_plan = env.plan
            best_step = env.count_step
            best_brick = env.count_brick
            best_tb = env.total_brick
            best_env = env.environment_memory
        iou_test_total+=iou_test
        reward_test_total+=reward_test
    reward_test_total=reward_test_total/N_iteration_test
    iou_test_total=iou_test_total/N_iteration_test
    secs = int(time.time() - start_time_test)
    mins = secs / 60
    secs = secs % 60
    iou_all_average += iou_test_total
    iou_all_min = min(iou_min,iou_all_min)
    total_reward=0
    state=env.reset()
    env_plan=env.input_plan
    print("total_brick:", env.total_brick)
    print('iou_his:',iou_history)
    print('iou_min:',min(iou_history))
    env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)
    plt.savefig(save_path + "Plan" + str(plan_choose) + '.png')
iou_all_average = iou_all_average/2
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()