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
pth_plan = '7456'
sys.path.append('../../../Env/1D/')
from DMP_Env_1D_dynamic_usedata_plan import deep_mobile_printing_1d1r
save_path = "./log/DRQN/plot/1D/Dynamic/"
load_path = "./log/DRQN/1D/Dynamic/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print('1D_Static')
print("device_using:", device)
iou_all_average = 0
iou_all_min = 1
env = deep_mobile_printing_1d1r(data_path="../../../Env/1D/data_1d_dynamic_sin_envplan_500_test.pkl", random_choose_paln=False)
######################
# hyper parameter
minibatch_size = 64
Lr = 0.00001
N_iteration = 6000
N_iteration_test = 200
alpha = 0.9
Replay_memory_size = 1000
Update_traget_period = 200
Action_dim = env.action_dim
State_dim = env.state_dim
hidden_state_dim = 256
Time_step = 20
UPDATE_FREQ = 5
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
    def __init__(self, out_size, hidden_size):
        super(Q_NET, self).__init__()
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.fc_1 = get_and_init_FC_layer(State_dim + env.plan_width, 64)
        self.fc_2 = get_and_init_FC_layer(64, 128)
        self.fc_3 = get_and_init_FC_layer(128, 128)
        self.rnn = nn.LSTM(128, hidden_size, num_layers=1, batch_first=True)
        self.adv = get_and_init_FC_layer(hidden_size, self.out_size)
        self.val = get_and_init_FC_layer(hidden_size, 1)
        self.relu = nn.ReLU()
    def forward(self, x, plan, bsize, time_step, hidden_state, cell_state):
        plan = plan.view(bsize * time_step, env.plan_width)
        x = x.view(bsize * time_step, State_dim)
        x = torch.cat((x, plan), dim=1)
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

class Memory():
    def __init__(self, memsize):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)
    def add_episode(self, epsiode):
        self.memory.append(epsiode)
    def get_batch(self, bsize, time_step):
        sampled_epsiodes = random.sample(self.memory, bsize)
        batch = []
        for episode in sampled_epsiodes:
            point = np.random.randint(0, len(episode) + 1 - time_step)
            batch.append(episode[point:point + time_step])
        return batch

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
        self.replaymemory = Memory(Replay_memory_size)
    def choose_action(self, s, plan, hidden_state, cell_state):
        state = torch.from_numpy(s).float().to(self.device)
        env_plan = torch.from_numpy(plan).float().to(self.device)
        model_out = self.Eval_net.forward(state, env_plan, bsize=1, time_step=1, hidden_state=hidden_state,
                                            cell_state=cell_state)
        out = model_out[0]
        action = int(torch.argmax(out[0]))
        hidden_state = model_out[1][0]
        cell_state = model_out[1][1]
        return action, hidden_state, cell_state

test_agent = DQN_AGNET(device)
log_path = load_path + "Eval_net_episode_"
test_agent.Eval_net.load_state_dict(torch.load(log_path + pth_plan + ".pth", map_location='cuda:0'))
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
    prev_state=state[0]
    env_plan=state[1]
    reward_test = 0
    hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)
    while True:
        action, hidden_state_next, cell_state_next = test_agent.choose_action(prev_state, env_plan, hidden_state,
                                                                                cell_state)
        state_next, r, done = env.step(action)
        reward_test += r
        if done:
            break
        prev_state = state_next[0]
        hidden_state, cell_state = hidden_state_next, cell_state_next
    iou_test = env.iou()
    iou_min = min(iou_min, iou_test)
    if iou_test > best_iou:
        best_iou = iou_test
        best_step = env.count_step
        best_brick = env.count_brick
        best_tb = env.total_brick
        best_env = env.environment_memory
        best_plan=env.plan
    iou_test_total += iou_test
    reward_test_total += reward_test

reward_test_total = reward_test_total / N_iteration_test
iou_test_total = iou_test_total / N_iteration_test
secs = int(time.time() - start_time_test)
mins = secs / 60
secs = secs % 60
env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,
            best_iou=best_iou,best_step=best_step,best_brick=best_brick,best_plan=best_plan)
iou_all_average += iou_test_total
iou_all_min = min(iou_min,iou_all_min)
plt.savefig(save_path+"Plan_test_"+str(plan_choose)+'.png')
iou_all_average = iou_all_average
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()
