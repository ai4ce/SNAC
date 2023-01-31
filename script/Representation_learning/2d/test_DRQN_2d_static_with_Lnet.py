import sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import time
import os
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_static_Lnet_test import deep_mobile_printing_2d1r
import model
import argparse
import matplotlib.pyplot as plt
import statistics


parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='device')
parser.add_argument('--plan_type', default=1, type=int, help='dense:0; sparse:1')
parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
parser.add_argument('--Random_seed', default=4, type=int, help='random seed')
save_path = "./model_dir_DRQN2d_withLnet_staticsparse/"
args = parser.parse_args()

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)
pth_number = 9000
seeds=args.Random_seed
set_seed(seeds)
Lr=args.lr
N_iteration_test = 500
PALN_CHOICE=args.plan_type  ##0: dense 1: sparse
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
PLAN_LIST=["dense","sparse"]
PLAN_NAME=PLAN_LIST[PALN_CHOICE]
env = deep_mobile_printing_2d1r(plan_choose=PALN_CHOICE)
hidden_state_dim = 256
Action_dim = env.action_dim
State_dim = env.state_dim
print("State_dim:", State_dim)
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
    nn.init.xavier_uniform_(
       li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li

class Q_NET(nn.Module):
    def __init__(self,out_size,hidden_size):
        super(Q_NET, self).__init__()       
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.fc_1 = get_and_init_FC_layer(State_dim+2, 64)
        self.fc_2 = get_and_init_FC_layer(64, 128)
        self.fc_3 = get_and_init_FC_layer(128, 128)
        self.rnn=nn.LSTM(128,hidden_size,num_layers=1,batch_first=True)
        self.adv  = get_and_init_FC_layer(hidden_size, self.out_size)
        self.val  = get_and_init_FC_layer(hidden_size, 1)
        self.relu = nn.ReLU()
    def forward(self,state,pose,bsize,time_step,hidden_state,cell_state):
        pose=pose.view(bsize*time_step,2)
        state=state.view(bsize*time_step,State_dim)
        x=torch.cat((state,pose),dim=1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        x = self.relu(x)
        x = x.view(bsize,time_step,128)
        lstm_out = self.rnn(x,(hidden_state,cell_state))
        out = lstm_out[0][:,time_step-1,:]
        h_n = lstm_out[1][0]
        c_n = lstm_out[1][1]
        adv_out = self.adv(out)
        val_out = self.val(out)
        qout = val_out.expand(bsize,self.out_size) + (adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(bsize,self.out_size))
        return qout, (h_n,c_n)
    def init_hidden_states(self,bsize):
        h = torch.zeros(1,bsize,self.hidden_size).float().to(device)
        c = torch.zeros(1,bsize,self.hidden_size).float().to(device)
        return h,c


class DQN_AGNET():
    def __init__(self,device):
        self.device=device
        self.Eval_net= Q_NET(Action_dim,hidden_size=hidden_state_dim).to(device)
        self.Target_net = Q_NET(Action_dim,hidden_size=hidden_state_dim).to(device)
        self.learn_step = 0                                     # counting the number of learning for update traget periodiclly 
                                                                # counting the transitions     
        self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
        self.loss = nn.SmoothL1Loss()
        self.loss_his=[]
        self.greedy_epsilon=0.0
    def choose_action(self, s, pose, hidden_state,cell_state):
        state=torch.from_numpy(s).float().to(self.device)
        pose = torch.FloatTensor(pose).unsqueeze(0).to(device)
        choose=np.random.uniform()
        if choose<=self.greedy_epsilon:
            model_out = self.Eval_net.forward(state, pose, bsize=1,time_step=1,hidden_state=hidden_state,cell_state=cell_state)
            action=np.random.randint(0, Action_dim)
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
        else:
            model_out = self.Eval_net.forward(state, pose, bsize=1,time_step=1,hidden_state=hidden_state,cell_state=cell_state)
            out = model_out[0]
            action = int(torch.argmax(out[0]))
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]    
        return action, hidden_state, cell_state

Lnet = model.SNAC_Lnet(input_size=105, hidden_size=128, device=device, Loss_type="L2")
model_checkpoint = torch.load("./model_dir_explict_representation_lstm_static_sparse/Lr_0.0001_batchsize_500/model_01000000.pth", map_location=device)
Lnet.load_state_dict(model_checkpoint)
Lnet.to(device)
Lnet.eval()

test_agent=DQN_AGNET(device)
test_agent.Eval_net.load_state_dict(torch.load(save_path+"DRQN_2d_static_withLnet_sparse_lr5e-05_seed_"+str(seeds)+"/Eval_net_episode_"+str(pth_number)+".pth" ,map_location=device))

best_iou=0
iou_test_total=0
reward_test_total=0
iou_min = 1
iou_history = []
start_time_test = time.time()
best_env = np.array([])
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)

for i in range(N_iteration_test):
    print(i)
    state = env.reset()
    current_pos = state[1]
    hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
    reward_test=0
    hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)
    while True:
        current_obs = state[0]
        current_pos = np.array(current_pos)
        torch_current_obs = torch.from_numpy(current_obs).float().unsqueeze(0).to(device)
        torch_current_pos = torch.from_numpy(current_pos).float().unsqueeze(0).to(device)
        torch_current_pos = torch_current_pos.unsqueeze(0)

        action, hidden_state_next, cell_state_next = test_agent.choose_action(current_obs,current_pos, hidden_state, cell_state)
        state_next, r, done = env.step(action)

        torch_action = torch.from_numpy(np.array(action).reshape(1,1,1)).float().to(device)
        next_obs = state_next[0]
        torch_next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)

        predicted_pos, hidden_next, cell_next =Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
        predict_pos = predicted_pos[0].squeeze().detach().cpu().numpy()
        predict_next_pos = [0,0]
        predict_next_pos[0] = round(predict_pos[0])  
        predict_next_pos[1] = round(predict_pos[1]) 

        reward_test += r
        if done:
            break
        state = state_next
        current_pos = predict_next_pos
        hidden_batch, cell_batch = hidden_next, cell_next
        hidden_state, cell_state = hidden_state_next, cell_state_next
    reward_test_total += reward_test
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
    iou_test_total += iou_test

iou_test_total = iou_test_total / N_iteration_test
secs = int(time.time() - start_time_test)
mins = secs / 60
secs = secs % 60
env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)


info_file = save_path+"Seed"+str(seeds)+"_lr"+str(Lr)
plt.savefig(info_file + '.png')
with open(info_file + ".pkl",'wb') as f:
    pickle.dump(iou_history, f)
print('iou_all_min:',iou_min)
print("iou_ave:",iou_test_total)
STD = statistics.stdev(iou_history)
print("std:",STD)
print("best IOU:",best_iou)



