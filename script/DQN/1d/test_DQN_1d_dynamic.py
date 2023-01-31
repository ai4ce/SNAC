import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import time
import os
from collections import deque
sys.path.append('../../../Env/1D/')
from DMP_Env_1D_dynamic_usedata_plan import deep_mobile_printing_1d1r

pth_plan = '0'
save_path = "./log/plot/1D/Dynamic/"
load_path = "./log/1D_Dynamic/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('1D_Dynamic')
print("device_using:", device)
iou_all_average = 0
iou_all_min = 1
env = deep_mobile_printing_1d1r(data_path="../../../Env/1D/data_1d_dynamic_sin_envplan_500_test.pkl", random_choose_paln=False)

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    li.weight.data.normal_(0, 0.1)
    return li

class Q_NET(nn.Module):
    def __init__(self, ):
        super(Q_NET, self).__init__()
        self.fc_1 = get_and_init_FC_layer(State_dim + 1 + plan_dim, 64)
        self.fc_2 = get_and_init_FC_layer(64, 128)
        self.fc_3 = get_and_init_FC_layer(128, 128)
        self.out = get_and_init_FC_layer(128, 1)
        self.relu = nn.ReLU()
    def forward(self, s, a, plan):
        x = torch.cat((s, a, plan), dim=1)
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
        state = torch.FloatTensor(s).unsqueeze(0).to(device)
        env_plan = torch.from_numpy(env_plan).float().to(device)
        env_plan = env_plan.unsqueeze(0)
        Q = []
        for item in range(Action_dim):
            a = torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
            Q_value = self.Eval_net(state, a, env_plan)
            Q.append(Q_value.item())
        action = np.argmax(Q)
        return action

test_agent = DQN_AGNET(device)
log_path = load_path + "Eval_net_episode_"
test_agent.Eval_net.load_state_dict(torch.load(log_path + pth_plan + ".pth", map_location='cpu'))
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
    state = env.reset()
    prev_state=state[0]
    env_plan=state[1]
    reward_test = 0
    while True: 
        action = test_agent.choose_action(prev_state, env_plan)
        state_next, r, done = env.step(action)
        reward_test += r
        if done:
            break
        prev_state = state_next[0]
    iou_test = env.iou()
    iou_min = min(iou_min, iou_test)
    if iou_test > best_iou:
        best_iou = iou_test
        best_step = env.count_step
        best_brick = env.count_brick
        best_env = env.environment_memory
        best_plan = env.plan
        best_tb = env.total_brick
    iou_test_total += iou_test
    reward_test_total += reward_test
reward_test_total = reward_test_total / N_iteration_test
iou_test_total = iou_test_total / N_iteration_test
secs = int(time.time() - start_time_test)
mins = secs / 60
secs = secs % 60
env.render(ax, iou_average=iou_test_total, iou_min=iou_min, iter_times=N_iteration_test, best_env=best_env,
            best_iou=best_iou, best_step=best_step, best_brick=best_brick,best_plan=best_plan)
iou_all_average += iou_test_total
iou_all_min = min(iou_min,iou_all_min)
plt.savefig(save_path+"Plan"+'.png')
iou_all_average = iou_all_average
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()
