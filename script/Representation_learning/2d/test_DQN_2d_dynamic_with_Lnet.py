import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_dynamic_usedata_plan import deep_mobile_printing_2d1r
import pickle
import matplotlib.pyplot as plt
import time
import os
import random
import model
import argparse
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', help='device')
parser.add_argument('--plan_type', default=0, type=int, help='dense:0; sparse:1')
parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
parser.add_argument('--Random_seed', default=4, type=int, help='random seed')
save_path = "./model_dir_DQN2d_withLnet_dynamicdense_croppedplan/"
args = parser.parse_args()

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)
pth_number = 9382
seeds=args.Random_seed
set_seed(seeds)
Lr=args.lr
N_iteration_test = 200
PALN_CHOICE=args.plan_type  ##0: dense 1: sparse
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
PLAN_LIST=["dense","sparse"]
PLAN_NAME=PLAN_LIST[PALN_CHOICE]
env_test = deep_mobile_printing_2d1r("../../../Env/2D/data_2d_dynamic_"+PLAN_NAME+"_envplan_500_test.pkl")
Action_dim = env_test.action_dim
State_dim = env_test.state_dim
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
        self.fc_1 = get_and_init_FC_layer(State_dim+49+1, 64)
        self.fc_2 = get_and_init_FC_layer(64, 128)
        self.fc_3 = get_and_init_FC_layer(128, 128)
        self.out = get_and_init_FC_layer(128, 1)
        self.relu = nn.ReLU()
    def forward(self, state, action,croppedplan,bsize):
        x=torch.cat((state, croppedplan, action),dim=1)
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
    def choose_action(self, s, croppedplan):      
        env_plan=torch.from_numpy(croppedplan).float().to(device)
        state = torch.FloatTensor(s).to(device)
        choose=np.random.uniform()
        Q=[]
        if choose<=self.greedy_epsilon:
            action=np.random.randint(0, Action_dim)         
        else:
            for item in range(Action_dim):
                a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                Q_value=self.Eval_net(state, a, env_plan, 1)
                Q.append(Q_value.item())  
            action=np.argmax(Q)    
        return action

Lnet = model.SNAC_Lnet(input_size=105, hidden_size=128, device=device, Loss_type="L2")
model_checkpoint = torch.load("./model_dir_explict_representation_lstm_dynamic_dense/Lr_0.0001_batchsize_500/model_01000000.pth", map_location=device)
Lnet.load_state_dict(model_checkpoint)
Lnet.to(device)
Lnet.eval()

test_agent=DQN_AGNET(device)
test_agent.Eval_net.load_state_dict(torch.load(save_path+"DQN_2d_dynamic_withLnet_dense_lr5e-05_seed_"+str(seeds)+"/Eval_net_episode_"+str(pth_number)+".pth" ,map_location=device))

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
    obs = env_test.reset()
    current_pos = obs[2]
    hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
    reward_test=0
    while True:
        current_obs = obs[0]
        current_cropped_plan = env_test.plan[
                    current_pos[0] - env_test.HALF_WINDOW_SIZE:current_pos[0] + env_test.HALF_WINDOW_SIZE + 1, \
                    current_pos[1] - env_test.HALF_WINDOW_SIZE:current_pos[1] + env_test.HALF_WINDOW_SIZE + 1]
        current_cropped_plan = current_cropped_plan.flatten().reshape(1, -1)
        current_pos = np.array(current_pos)
        torch_current_obs = torch.from_numpy(current_obs).float().unsqueeze(0).to(device)
        torch_current_pos = torch.from_numpy(current_pos).float().unsqueeze(0).to(device)
        torch_current_pos = torch_current_pos.unsqueeze(0)

        action = test_agent.choose_action(current_obs, current_cropped_plan)

        new_obs, reward, done = env_test.step(action)

        torch_action = torch.from_numpy(np.array(action).reshape(1,1,1)).float().to(device)
        next_pos = new_obs[2]
        next_obs = new_obs[0]
        torch_next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)

        predicted_pos, hidden_next, cell_next =Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
        predict_pos = predicted_pos[0].squeeze().detach().cpu().numpy()
        predict_next_pos = [0,0]
        predict_next_pos[0] = round(predict_pos[0])  
        predict_next_pos[1] = round(predict_pos[1]) 
        predict_next_pos = env_test.clip_position(predict_next_pos)

        if done:
            break
        obs = new_obs
        current_pos = predict_next_pos
        hidden_batch, cell_batch = hidden_next, cell_next
    iou_test = iou(env_test.environment_memory,env_test.plan,env_test.HALF_WINDOW_SIZE,env_test.plan_height,env_test.plan_width)
    iou_history.append(iou_test)
    iou_min = min(iou_min,iou_test)
    if iou_test > best_iou:
        best_iou = iou_test
        best_plan = env_test.plan
        best_step = env_test.count_step
        best_brick = env_test.count_brick
        best_tb = env_test.total_brick
        best_env = env_test.environment_memory
        best_plan_index = env_test.index_random
    iou_test_total += iou_test

iou_test_total = iou_test_total / N_iteration_test
secs = int(time.time() - start_time_test)
mins = secs / 60
secs = secs % 60
env_test.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env, \
                best_iou=best_iou,best_step=best_step,best_brick=best_brick,best_plan=best_plan)


info_file = save_path+"Seed"+str(seeds)+"_lr"+str(Lr)
plt.savefig(info_file + 'demo.png')
with open(info_file + ".pkl",'wb') as f:
    pickle.dump(iou_history, f)
print('iou_all_min:',iou_min)
print("iou_ave:",iou_test_total)
STD = statistics.stdev(iou_history)
print("std:",STD)
print("best IOU:",best_iou)
print("best_plan_index:",best_plan_index)
