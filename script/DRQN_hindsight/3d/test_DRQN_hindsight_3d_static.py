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

pth_plan = ['3', '135']

sys.path.append('../../Env/3D/')
from DMP_simulator_3d_static_circle_test import deep_mobile_printing_3d1r

save_path = "./log/DRQN_hindsight/plot/3D/Static/"
load_path = "./log/DRQN_hindsight/3D/Static/plan_"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('3D_Static')

iou_all_average = 0
iou_all_min = 1

for plan_choose in range(2):
    env = deep_mobile_printing_3d1r(plan_choose=plan_choose)
    ######################
    # hyper parameter
    minibatch_size = 64
    Lr = 0.0001
    N_iteration = 10000
    N_iteration_test = 50
    alpha = 0.9
    Replay_memory_size = 1000
    Update_traget_period = 200
    Action_dim = env.action_dim
    State_dim = env.state_dim
    hidden_state_dim = 256
    Time_step = 20
    UPDATE_FREQ = 5
    INITIAL_EPSILON = 0.2
    FINAL_EPSILON = 0.0
    ######################
    env.reset()
    print("State_dim:",State_dim)
    print("total_step:",env.total_step)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^")



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
        def __init__(self, out_size, hidden_size):
            super(Q_NET, self).__init__()

            self.out_size = out_size
            self.hidden_size = hidden_size

           

            self.fc_1 = get_and_init_FC_layer(State_dim, 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.rnn = nn.LSTM(128 , hidden_size, num_layers=1, batch_first=True)
            self.adv = get_and_init_FC_layer(hidden_size, self.out_size)
            self.val = get_and_init_FC_layer(hidden_size, 1)
            self.relu = nn.ReLU()

        def forward(self, x, bsize, time_step, hidden_state, cell_state):
            

            x = x.view(bsize * time_step, State_dim)

            x = self.fc_1(x)
            x = self.relu(x)
            x = self.fc_2(x)
            x = self.relu(x)
            x = self.fc_3(x)
            x = self.relu(x)
           
            x = x.view(bsize, time_step, 128)
            lstm_out = self.rnn(x, (hidden_state, cell_state))
            out = lstm_out[0][:, time_step - 1, :]
            h_n = lstm_out[1][0]
            c_n = lstm_out[1][1]
            adv_out = self.adv(out)
            val_out = self.val(out)
            qout = val_out.expand(bsize, self.out_size) + (
                        adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(bsize, self.out_size))
            return qout, (h_n, c_n)

        def init_hidden_states(self, bsize):
            h = torch.zeros(1, bsize, self.hidden_size).float().to(device)
            c = torch.zeros(1, bsize, self.hidden_size).float().to(device)
            return h, c


    


    class DQN_AGNET():
        def __init__(self, device):
            self.device = device
            self.Eval_net = Q_NET(Action_dim, hidden_size=hidden_state_dim).to(device)
            self.Target_net = Q_NET(Action_dim, hidden_size=hidden_state_dim).to(device)
            self.learn_step = 0  # counting the number of learning for update traget periodiclly
            # counting the transitions
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
            self.loss = nn.SmoothL1Loss()
            self.loss_his = []
            self.greedy_epsilon = 0.2

        def choose_action(self, s, hidden_state, cell_state):

            state = torch.from_numpy(s).float().to(self.device)
          
            model_out = self.Eval_net.forward(state, bsize=1, time_step=1, hidden_state=hidden_state,
                                              cell_state=cell_state)
            out = model_out[0]
            action = int(torch.argmax(out[0]))
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]

            return action, hidden_state, cell_state

        


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
        reward_test=0
        print('Ep:',i)

        hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)

        while True:
            action, hidden_state_next, cell_state_next = test_agent.choose_action(state, hidden_state, cell_state)
            state_next, r, done = env.step(action)
            reward_test += r
            if done:
                break
            state = state_next
            hidden_state, cell_state = hidden_state_next, cell_state_next
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
    hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)
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