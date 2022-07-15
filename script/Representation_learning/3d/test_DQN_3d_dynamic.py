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
sys.path.append('../../Env/3D/')
pth_plan = ['3', '135']

sys.path.append('3D/Static/')
from DMP_simulator_3d_static_circle_test import deep_mobile_printing_3d1r

save_path = "./log/DQN/plot/3D/Dynamic/"
load_path = "./log/DQN/3D/Dynamic/plan_"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('3D_Dynamic')

iou_all_average = 0
iou_all_min = 1

for plan_choose in range(2):
    env = deep_mobile_printing_3d1r(plan_choose=plan_choose)
    ######################
    # hyper parameter
    minibatch_size = 2000
    Lr = 0.001
    N_iteration = 3000
    N_iteration_test = 10
    alpha = 0.9
    Replay_memory_size = 50000
    Update_traget_period = 200
    UPDATE_FREQ = 1

    Action_dim = env.action_dim
    State_dim = env.state_dim
    plan_dim = env.plan_width
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0

    print("state_dim", State_dim)
    print("total_step", env.total_step)
    print("Action_dim", Action_dim)


    def get_and_init_FC_layer(din, dout):
        li = nn.Linear(din, dout)
        li.weight.data.normal_(0, 0.1)
        return li


    class Q_NET(nn.Module):
        def __init__(self, ):
            super(Q_NET, self).__init__()

            self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                                         stride=2)  # potential check - in_channels
            self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
            self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2)
            self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2)

            self.fc_1 = get_and_init_FC_layer(State_dim + 1 + 32, 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.out = get_and_init_FC_layer(128, 1)
            self.relu = nn.ReLU()

        def forward(self, s, a, plan, bsize):
            plan = plan.view(bsize, 1, 20, 20)
            conv_out = self.conv_layer1(plan)
            conv_out = self.relu(conv_out)
            conv_out = self.conv_layer2(conv_out)
            conv_out = self.relu(conv_out)

            conv_out = self.conv_layer3(conv_out)
            conv_out = self.relu(conv_out)
            conv_out = conv_out.view(bsize, 32)

            x = torch.cat((s, a, conv_out), dim=1)
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
            self.replay_memory.append((s, a, r, s_next, env_plan))
            self.count_memory += 1

        def choose_action(self, s, env_plan):

            state = torch.FloatTensor(s).to(device)
            env_plan = torch.from_numpy(env_plan).float().to(device)
            env_plan = env_plan.unsqueeze(0)
            Q = []
            for item in range(Action_dim):
                a = torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)

                Q_value = self.Eval_net(state, a, env_plan, 1)
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
            env_plans = []

            for b in minibatch:
                current_states.append(b[0])
                acts.append(b[1])
                rewards.append(b[2])
                next_states.append(b[3])
                env_plans.append(b[4])

            current_states = np.array(current_states)
            # print("current_states_shape",current_states.shape)
            acts = np.array(acts).reshape(minibatch_size, 1)
            # print("action_shape",acts.shape)
            rewards = np.array(rewards).reshape(minibatch_size, 1)
            next_states = np.array(next_states)
            env_plans = np.array(env_plans)

            b_state = torch.from_numpy(current_states).float().to(device).squeeze(1)
            b_action = torch.from_numpy(acts).float().to(device)
            b_reward = torch.from_numpy(rewards).float().to(device)
            b_state_next = torch.from_numpy(next_states).float().to(device).squeeze(1)
            b_env_plans = torch.from_numpy(env_plans).float().to(device)

            Q_eval = self.Eval_net(b_state, b_action, b_env_plans, minibatch_size)
            Q = torch.zeros(minibatch_size, Action_dim).to(device)
            for item in range(Action_dim):
                a = torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                a = a.expand(minibatch_size, -1)
                Q_value = self.Target_net(b_state_next, a, b_env_plans, minibatch_size).detach().squeeze(1)
                Q[:, item] = Q_value
            Q_next = torch.max(Q, dim=1)[0].reshape(minibatch_size, 1)
            Q_target = b_reward + alpha * Q_next
            loss = self.loss(Q_eval, Q_target)

            loss.backward()
            self.optimizer.step()
            self.learn_step += 1
            self.loss_his.append(loss.item())


    test_agent=DQN_AGNET(device)
    log_path = load_path + str(plan_choose) + "/" + "Eval_net_episode_"
    test_agent.Eval_net.load_state_dict(torch.load(log_path + pth_plan[plan_choose]+".pth" ,map_location='cuda:0'))
    best_iou=0
    iou_test_total=0
    reward_test_total=0
    iou_min = 1
    iou_history = []
    start_time_test = time.time()
    best_env = np.array([])
    fig = plt.figure(figsize=[10, 5])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    for i in range(N_iteration_test):
        state = env.reset()
        prev_state=state[0]
        env_plan=state[1]
        reward_test=0
        print('Ep:',i)
        while True:
            action = test_agent.choose_action(prev_state,env_plan)
            state_next, r, done = env.step(action)
            reward_test += r
            if done:
                break
            prev_state = state_next[0]
        iou_test=env.iou()
        iou_history.append(iou_test)

        iou_min = min(iou_min,iou_test)
        if iou_test > best_iou:
            best_iou = iou_test
            best_plan = env.plan
            best_step = env.count_step
            best_brick = env.count_brick
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

    total_reward=0
    state=env.reset()
    env_plan=env.input_plan
    print("total_brick:", env.total_brick)
    print('iou_his:',iou_history)
    print('iou_min:',min(iou_history))
    # env.render(ax)
    if best_iou>0:
        env.render(ax1,ax2,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,
                   best_iou=best_iou,best_step=best_step,best_brick=best_brick,position_memo = best_posi)
    else:
        env.render(ax1,ax2,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)
    plt.savefig(save_path + "Plan" + str(plan_choose) + '.png')


    # while True:
    #     action,hidden_state_next, cell_state_next = test_agent.choose_action(state,env_plan,hidden_state, cell_state)
    #     state_next, r, done = env.step(action)
    #     # env.render(ax,iou_min,iou_test_total,N_iteration_test)
    #     # plt.pause(0.1)
    #     total_reward+=r
    #     if done:
    #         env.render(ax, iou_min, iou_test_total, N_iteration_test)
    #         plt.savefig(save_path + "Plan" + str(plan_choose) + '.png')
    #         break
    #     state = state_next
    #     hidden_state, cell_state = hidden_state_next, cell_state_next

iou_all_average = iou_all_average/2
print('#### Finish #####')
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()