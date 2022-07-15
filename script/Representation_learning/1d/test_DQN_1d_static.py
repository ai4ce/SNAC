import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import pickle
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from collections import deque
from DMP_Env_1D_static_test import deep_mobile_printing_1d1r

pth_plan = ['0', '0', '0']

save_path = "./log/plot/1D/Static/"
load_path = "./log/1D_Static/plan_"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('1D_Static')

iou_all_average = 0
iou_all_min = 1
for plan_choose in range(3):

    env = deep_mobile_printing_1d1r(plan_choose=plan_choose)
    ######################
    # hyper parameter
    minibatch_size = 2000
    Lr = 0.001
    N_iteration = 3000
    N_iteration_test = 10
    alpha = 0.9
    Replay_memory_size = 50000
    Update_traget_period = 200
    Action_dim = env.action_dim
    State_dim = env.state_dim
    UPDATE_FREQ = 1
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    ######################
    PLAN_DIM = env.plan_width
    env.reset()
    print("State_dimension:", State_dim)
    print("Total Brick: ", env.total_brick)
    print("plan_width", PLAN_DIM)
    print('######################')


    def get_and_init_FC_layer(din, dout):
        li = nn.Linear(din, dout)
        li.weight.data.normal_(0, 0.1)
        return li


    class Q_NET(nn.Module):
        def __init__(self, ):
            super(Q_NET, self).__init__()
            self.fc_1 = get_and_init_FC_layer(State_dim + 1, 64)
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
            # self.replay_memory = np.zeros((Replay_memory_size, State_dim * 2 + 2))
            self.replay_memory = deque(maxlen=Replay_memory_size)
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
            self.greedy_epsilon = 0.2
            self.loss = nn.SmoothL1Loss()
            self.loss_his = []

        def store_memory(self, s, a, r, s_next):
            # transition=np.concatenate((s,[a],[r],s_next))
            # index=self.count_memory%Replay_memory_size
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

        def learning_process(self):
            self.optimizer.zero_grad()
            self.Eval_net.train()
            if self.learn_step % Update_traget_period == 0:
                self.Target_net.load_state_dict(self.Eval_net.state_dict())

            minibatch = random.sample(self.replay_memory, minibatch_size)

            current_states = []
            acts = []
            rewards = []
            next_states = []

            for b in minibatch:
                current_states.append(b[0])
                acts.append(b[1])
                rewards.append(b[2])
                next_states.append(b[3])

            current_states = np.array(current_states)

            acts = np.array(acts).reshape(minibatch_size, 1)

            rewards = np.array(rewards).reshape(minibatch_size, 1)
            next_states = np.array(next_states)

            b_state = torch.from_numpy(current_states).float().to(device)
            b_action = torch.from_numpy(acts).float().to(device)
            b_reward = torch.from_numpy(rewards).float().to(device)

            b_state_next = torch.from_numpy(next_states).float().to(device)

            Q_eval = self.Eval_net(b_state, b_action)

            Q = torch.zeros(minibatch_size, Action_dim).to(device)
            for item in range(Action_dim):
                a = torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                a = a.expand(minibatch_size, -1)
                Q_value = self.Target_net(b_state_next, a).detach().squeeze(1)
                Q[:, item] = Q_value
            Q_next = torch.max(Q, dim=1)[0].reshape(minibatch_size, 1)
            Q_target = b_reward + alpha * Q_next

            loss = self.loss(Q_eval, Q_target)

            loss.backward()
            self.optimizer.step()
            self.learn_step += 1
            self.loss_his.append(loss.item())


    test_agent = DQN_AGNET(device)
    log_path = load_path + str(plan_choose) + "/" + "Eval_net_episode_"

    test_agent.Eval_net.load_state_dict(torch.load(log_path + pth_plan[plan_choose] + ".pth", map_location='cpu'))
    test_agent.greedy_epsilon = 0.08
    best_iou = 0
    best_env = np.array([])
    iou_test_total = 0
    iou_min = 1
    reward_test_total = 0
    start_time_test = time.time()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    for ep in range(N_iteration_test):
        print(ep)
        state = env.reset()
        reward_test = 0

        while True:
            state = state.reshape(state.shape[-1])
            action = test_agent.choose_action(state)
            state_next, r, done = env.step(action)
            state_next = state_next.reshape(state_next.shape[-1])
            reward_test += r
            if done:
                break
            state = state_next

        iou_test = env.iou()
        iou_min = min(iou_min, iou_test)

        if iou_test > best_iou:
            best_iou = iou_test
            best_plan = env.plan
            best_step = env.count_step
            best_brick = env.count_brick
            best_env = env.environment_memory
            best_tb = env.total_brick
        iou_test_total += iou_test
        reward_test_total += reward_test

    reward_test_total = reward_test_total / N_iteration_test
    iou_test_total = iou_test_total / N_iteration_test
    secs = int(time.time() - start_time_test)
    mins = secs / 60
    secs = secs % 60
    env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)

    iou_all_average += iou_test_total
    iou_all_min = min(iou_min,iou_all_min)
    plt.savefig(save_path+"Plan"+str(plan_choose)+'.png')

iou_all_average = iou_all_average/3
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()
