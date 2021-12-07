import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import time
import os
from collections import deque
sys.path.append('../../../Env/1D/')
from DMP_Env_1D_static import deep_mobile_printing_1d1r
from DMP_Env_1D_static_hindsight_replay import deep_mobile_printing_1d1r_hindsight
def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)
######################
# hyper parameter
seeds=5
set_seed(seeds)
minibatch_size = 64
Lr = 0.0001
N_iteration = 8000
N_iteration_test = 10
alpha = 0.9
Replay_memory_size = 1000
Update_traget_period = 200
hidden_state_dim = 256
Time_step = 15
UPDATE_FREQ = 5
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0
use_hindsight=True
######################
# 0 Sin, 1 Gaussian, 2 Step
PALN_CHOICE=2  # 0 Sin, 1 Gaussian, 2 Step
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PLAN_LIST=["sin","Gaussian","step"]
PLAN_NAME=PLAN_LIST[PALN_CHOICE]
OUT_FILE_NAME="DRQN_hindsight_1d_"+PLAN_NAME+"_lr"+str(Lr)+"_seed_"+str(seeds)
print(OUT_FILE_NAME)
log_path="/mnt/NAS/home/WenyuHan/SNAC/DRQN_Hindsight/1D/log/static/"+OUT_FILE_NAME+"/"
env = deep_mobile_printing_1d1r(plan_choose=PALN_CHOICE)
env_hindsight = deep_mobile_printing_1d1r_hindsight(PALN_CHOICE)
print("device_using:", device)
Action_dim = env.action_dim
State_dim = env.state_dim
print("State_dimension", State_dim)
print("plan_width", env.plan_width)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
if os.path.exists(log_path) == False:
    os.makedirs(log_path)

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(
       li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li

class Q_NET(nn.Module):
    def __init__(self, out_size, hidden_size):
        super(Q_NET, self).__init__()
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.fc_1 = get_and_init_FC_layer(State_dim, 64)
        self.fc_2 = get_and_init_FC_layer(64, 128)
        self.fc_3 = get_and_init_FC_layer(128, 128)
        self.rnn = nn.LSTM(128, hidden_size, num_layers=1, batch_first=True)
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
    def init_hidden_states(self,bsize):
        h = torch.zeros(1,bsize,self.hidden_size).float().to(device)
        c = torch.zeros(1,bsize,self.hidden_size).float().to(device)
        return h,c

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
        self.learn_step = 0
        # counting the number of learning for update target periodically
        # counting the transitions
        self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
        self.loss = nn.SmoothL1Loss()
        self.loss_his = []
        self.greedy_epsilon = 0.2
        self.replaymemory = Memory(Replay_memory_size)
    def choose_action(self, s, hidden_state, cell_state):
        state = torch.from_numpy(s).float().to(self.device)
        choose = np.random.uniform()
        if choose<=self.greedy_epsilon:
            model_out = self.Eval_net.forward(state,bsize=1,time_step=1,hidden_state=hidden_state,cell_state=cell_state)
            action=np.random.randint(0, Action_dim)
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
        else:
            model_out = self.Eval_net.forward(state,bsize=1,time_step=1,hidden_state=hidden_state,cell_state=cell_state)
            out = model_out[0]
            action = int(torch.argmax(out[0]))
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
        return action, hidden_state, cell_state
    def learning_process(self):
        self.optimizer.zero_grad()
        self.Eval_net.train()
        if self.learn_step % Update_traget_period == 0:
            self.Target_net.load_state_dict(self.Eval_net.state_dict())
        hidden_batch, cell_batch = self.Eval_net.init_hidden_states(bsize=minibatch_size)
        batch = self.replaymemory.get_batch(bsize=minibatch_size, time_step=Time_step)
        current_states = []
        acts = []
        rewards = []
        next_states = []
        for b in batch:
            cs, ac, rw, ns = [], [], [], []
            for element in b:
                cs.append(element[0])
                ac.append(element[1])
                rw.append(element[2])
                ns.append(element[3])
            current_states.append(cs)
            acts.append(ac)
            rewards.append(rw)
            next_states.append(ns)
        current_states = np.array(current_states)
        acts = np.array(acts)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        torch_current_states = torch.from_numpy(current_states).float().to(self.device)
        torch_acts = torch.from_numpy(acts).long().to(self.device)
        torch_rewards = torch.from_numpy(rewards).float().to(self.device)
        torch_next_states = torch.from_numpy(next_states).float().to(self.device)
        Q_s, _ = self.Eval_net.forward(torch_current_states, bsize=minibatch_size, time_step=Time_step,
                                       hidden_state=hidden_batch, cell_state=cell_batch)
        Q_s_a = Q_s.gather(dim=1, index=torch_acts[:, Time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)
        Q_next, _ = self.Target_net.forward(torch_next_states, bsize=minibatch_size, time_step=Time_step,
                                            hidden_state=hidden_batch, cell_state=cell_batch)
        Q_next_max, __ = Q_next.detach().max(dim=1)
        target_values = torch_rewards[:, Time_step - 1] + (alpha * Q_next_max)
        loss = self.loss(Q_s_a, target_values)
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1
        train_loss=loss.item()
        return train_loss

#### initial fill the replaymemory
agent = DQN_AGNET(device)
for i in range(0, Replay_memory_size):
    prev_state = env.reset()
    local_memory = []
    while True:
        action = np.random.randint(0, Action_dim)
        next_state, reward, done = env.step(action)
        local_memory.append((prev_state, action, reward, next_state))
        prev_state = next_state
        if done:
            break
    agent.replaymemory.add_episode(local_memory)
agent.greedy_epsilon = INITIAL_EPSILON
print("agent greedy_epsilon", agent.greedy_epsilon)
best_reward = -500
total_steps = 0
writer = SummaryWriter('/mnt/NAS/home/WenyuHan/SNAC/DRQN_hindsight/1D/DRQN_hindsight_1d_static')

for episode in range(N_iteration):
    state = env.reset()
    print("plan:", env.one_hot)
    print("total_brick", env.total_brick)
    reward_train = 0
    step_size_memory=[]
    start_time = time.time()
    local_memory = []
    hidden_state, cell_state = agent.Eval_net.init_hidden_states(bsize=1)
    while True:
        total_steps += 1
        action, hidden_state_next, cell_state_next = agent.choose_action(state, hidden_state, cell_state)
        state_next, r, done = env.step(action)
        step_size_memory.append(env.step_size)
        local_memory.append((state, action, r, state_next))
        reward_train += r
        if total_steps % UPDATE_FREQ == 0:
            train_loss=agent.learning_process()
        if done:
            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            break
        state = state_next
        hidden_state, cell_state = hidden_state_next, cell_state_next
    agent.replaymemory.add_episode(local_memory)
    train_iou=env.iou()
    #########  hindsight 
    if use_hindsight:
        local_memory_hindsight=[]
        _ = env_hindsight.reset()
        env_hindsight.plan=env.environment_memory[0,env.HALF_WINDOW_SIZE:env.HALF_WINDOW_SIZE+env.plan_width]
        for i,element in enumerate(local_memory):
            _, r, _ = env_hindsight.step(element[1],step_size_memory[i])
            local_memory_hindsight.append((element[0],element[1],r,element[3]))
        agent.replaymemory.add_episode(local_memory_hindsight)   
    ############ test agent
    iou_test = 0
    reward_test_total = 0
    for _ in range(N_iteration_test):
        state = env.reset()
        reward_test = 0
        hidden_state, cell_state = agent.Eval_net.init_hidden_states(bsize=1)
        while True:
            action, hidden_state_next, cell_state_next = agent.choose_action(state, hidden_state, cell_state)
            state_next, r, done = env.step(action)
            reward_test += r
            if done:
                break
            state = state_next
            hidden_state, cell_state = hidden_state_next, cell_state_next
        reward_test_total += reward_test
        iou_test += env.iou()
    reward_test_total = reward_test_total / N_iteration_test
    IOU_test_total= iou_test/N_iteration_test
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60
    print('Epodise: ', episode,
          '| Ep_reward_test:', reward_test_total, '| Ep_IOU_test: ', IOU_test_total)
    print(" | time in %d minutes, %d seconds\n" % (mins, secs))
    if agent.greedy_epsilon > FINAL_EPSILON:
        agent.greedy_epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / N_iteration
    if reward_test_total >= best_reward:
        torch.save(agent.Eval_net.state_dict(), log_path + 'Eval_net_episode_%d.pth' % (episode))
        torch.save(agent.Target_net.state_dict(), log_path + 'Target_net_episode_%d.pth' % (episode))
        best_reward = reward_test_total
    writer.add_scalars(OUT_FILE_NAME,
                                   {'train_loss': train_loss,
                                    'train_reward': reward_train,
                                    'train_iou': train_iou,
                                   'test_reward': reward_test_total,
                                   'test_iou':IOU_test_total,}, episode)
JSON_log_PATH="./JSON/"
if os.path.exists(JSON_log_PATH)==False:
    os.makedirs(JSON_log_PATH)                                   
writer.export_scalars_to_json(JSON_log_PATH+OUT_FILE_NAME+".json")
writer.close()

