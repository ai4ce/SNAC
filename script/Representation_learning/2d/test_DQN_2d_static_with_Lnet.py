import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_static_Lnet_test import deep_mobile_printing_2d1r
import pickle
import matplotlib.pyplot as plt
import time
import os
import random
import model
import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:2', help='device')
parser.add_argument('--plan_type', default=1, type=int, help='dense:0; sparse:1')
parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--Random_seed', default=4, type=int, help='random seed')
save_path = "./model_dir_DQN2d_withLnet_staticsparse/"
args = parser.parse_args()

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)
pth_number = 9690
seeds=args.Random_seed
set_seed(seeds)
Lr=args.lr
N_iteration_test = 500
PALN_CHOICE=args.plan_type  ##0: dense 1: sparse
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
PLAN_LIST=["dense","sparse"]
PLAN_NAME=PLAN_LIST[PALN_CHOICE]
env = deep_mobile_printing_2d1r(plan_choose=PALN_CHOICE)
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
    def __init__(self, ):
        super(Q_NET, self).__init__()
        self.fc_1 = get_and_init_FC_layer(State_dim+3, 64)
        self.fc_2 = get_and_init_FC_layer(64, 128)
        self.fc_3 = get_and_init_FC_layer(128, 128)
        self.out = get_and_init_FC_layer(128, 1)
        self.relu = nn.ReLU()
    def forward(self, s, pose, a):
        x=torch.cat((s,pose,a),dim=1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        x = self.relu(x)
        Q = self.out(x)
        return Q

class DQN_AGNET():
    def __init__(self,device):
        self.Eval_net= Q_NET().to(device)
        self.Target_net = Q_NET().to(device)
        self.learn_step = 0                                     # counting the number of learning for update traget periodiclly 
        self.count_memory = 0                                         # counting the transitions 
        self.greedy_epsilon=0.0
    def choose_action(self, s, pose):      
        state = torch.FloatTensor(s).to(device)
        pose = torch.FloatTensor(pose).unsqueeze(0).to(device)
        choose=np.random.uniform()
        Q=[]
        if choose<=self.greedy_epsilon:
            action=np.random.randint(0, Action_dim)         
        else:
            for item in range(Action_dim):
                a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                Q_value=self.Eval_net(state, pose, a)
                Q.append(Q_value.item())  
            action=np.argmax(Q)    
        return action


Lnet = model.SNAC_Lnet(input_size=105, hidden_size=128, device=device, Loss_type="L2")
model_checkpoint = torch.load("./model_dir_explict_representation_lstm_static_sparse/Lr_0.0001_batchsize_500/model_01000000.pth", map_location=device)
Lnet.load_state_dict(model_checkpoint)
Lnet.to(device)
Lnet.eval()

test_agent=DQN_AGNET(device)
test_agent.Eval_net.load_state_dict(torch.load(save_path+"DQN_2d_static_withLnet_sparse_lr1e-05_seed_"+str(seeds)+"/Eval_net_episode_"+str(pth_number)+".pth" ,map_location=device))

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
    obs = env.reset()
    current_pos = obs[1]
    hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
    reward_test=0
    while True:
        current_obs = obs[0]
        current_pos = np.array(current_pos)
        torch_current_obs = torch.from_numpy(current_obs).float().unsqueeze(0).to(device)
        torch_current_pos = torch.from_numpy(current_pos).float().unsqueeze(0).to(device)
        torch_current_pos = torch_current_pos.unsqueeze(0)

        action = test_agent.choose_action(current_obs, current_pos)

        new_obs, reward, done = env.step(action)

        torch_action = torch.from_numpy(np.array(action).reshape(1,1,1)).float().to(device)
        next_pos = new_obs[1]
        next_obs = new_obs[0]
        torch_next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)

        predicted_pos, hidden_next, cell_next =Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
        predict_pos = predicted_pos[0].squeeze().detach().cpu().numpy()
        predict_next_pos = [0,0]
        predict_next_pos[0] = round(predict_pos[0])  
        predict_next_pos[1] = round(predict_pos[1]) 

        if done:
            break
        obs = new_obs
        current_pos = predict_next_pos
        hidden_batch, cell_batch = hidden_next, cell_next
    iou_test = iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
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
