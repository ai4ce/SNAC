import sys
import torch
import torch.nn as nn
import numpy as np
import random
import time
import os
import argparse
sys.path.append('../../../Env/2D/')
import model
from DMP_Env_2D_dynamic_usedata_plan import deep_mobile_printing_2d1r
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn import datasets, linear_model


def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:2', help='device')
parser.add_argument('--plan_type', default=1, type=int, help='dense:0; sparse:1')
parser.add_argument('--lr', default=0.000001, type=float, help='learning rate')
parser.add_argument('--Random_seed', default=1, type=int, help='random seed')
save_path = "./model_dir_DRQN2d_withLnet_croppedplan_dynamicsparse/"
args = parser.parse_args()

pth_number = 9889
seeds=args.Random_seed
set_seed(seeds)
Lr=args.lr
N_iteration_test = 500
PALN_CHOICE=args.plan_type  ##0: dense 1: sparse
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
PLAN_LIST=["dense","sparse"]
PLAN_NAME=PLAN_LIST[PALN_CHOICE]
env_test = deep_mobile_printing_2d1r("../../../Env/2D/data_2d_dynamic_"+PLAN_NAME+"_envplan_500_test.pkl")
hidden_state_dim = 256
Action_dim = env_test.action_dim
State_dim = env_test.state_dim
print("State_dim:", State_dim)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^")




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
        self.fc_1 = get_and_init_FC_layer(State_dim+49, 128)
        self.fc_2 = get_and_init_FC_layer(128, 256)
        self.fc_3 = get_and_init_FC_layer(256, 512)
        self.rnn = nn.LSTM(512, hidden_size, num_layers=1, batch_first=True)
        self.adv = get_and_init_FC_layer(hidden_size, self.out_size)
        self.val = get_and_init_FC_layer(hidden_size, 1)
        self.relu = nn.ReLU()
    def forward(self, state,cropped_plan, bsize, time_step, hidden_state, cell_state):
        state = state.view(bsize * time_step, State_dim)
        cropped_plan = cropped_plan.view(bsize * time_step, 49)
        input_data = torch.cat((state[:,:49],cropped_plan,state[:,49:51]),dim=1)
        x = self.fc_1(input_data)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        x = self.relu(x)
        x = x.view(bsize, time_step, 512)
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
        self.greedy_epsilon = 0.0
        
    def choose_action(self, s, croppedplan, hidden_state, cell_state):
        state = torch.from_numpy(s).float().to(self.device)
        env_plan = torch.from_numpy(croppedplan).float().to(self.device)
        choose = np.random.uniform()
        if choose <= self.greedy_epsilon:
            model_out = self.Eval_net.forward(state, env_plan, bsize=1, time_step=1, hidden_state=hidden_state,
                                              cell_state=cell_state)
            action = np.random.randint(0, Action_dim)
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
        else:
            model_out = self.Eval_net.forward(state, env_plan, bsize=1, time_step=1, hidden_state=hidden_state,
                                              cell_state=cell_state)
            out = model_out[0]
            action = int(torch.argmax(out[0]))
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
        return action, hidden_state, cell_state


def iou(environment_memory, environment_plan, HALF_WINDOW_SIZE, plan_height, plan_width):
    component1 = environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE + plan_height, \
                 HALF_WINDOW_SIZE:HALF_WINDOW_SIZE + plan_width].astype(bool)
    component2 = environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE + plan_height, \
                 HALF_WINDOW_SIZE:HALF_WINDOW_SIZE + plan_width].astype(bool)
    overlap = component1 * component2  # Logical AND
    union = component1 + component2  # Logical OR
    IOU = overlap.sum() / float(union.sum())
    return IOU

Lnet = model.SNAC_Lnet(input_size=105, hidden_size=128, device=device, Loss_type="L2")
model_checkpoint = torch.load("./model_dir_explict_representation_lstm_dynamic_sparse/Lr_5e-05_batchsize_500/model_01000000.pth", map_location=device)
Lnet.load_state_dict(model_checkpoint)
Lnet.to(device)
Lnet.eval()

test_agent=DQN_AGNET(device)
test_agent.Eval_net.load_state_dict(torch.load(save_path+"DRQN_2d_dynamic_withLnet_sparse_lr1e-06_seed_"+str(seeds)+"/Eval_net_episode_"+str(pth_number)+".pth" ,map_location=device))

best_iou=0
iou_test_total=0
reward_test_total=0
iou_min = 1
iou_history = []
distance_error_history = []
start_time_test = time.time()
best_env = np.array([])
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)

for i in range(N_iteration_test):
    print(i)
    state = env_test.reset()
    current_pos = state[2]
    hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
    env_plan=state[1]
    reward_test = 0
    distance_error = 0
    number_step = 0
    hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)
    while True:
        current_obs = state[0]
        current_cropped_plan = env_test.plan[
                    current_pos[0] - env_test.HALF_WINDOW_SIZE:current_pos[0] + env_test.HALF_WINDOW_SIZE + 1, \
                    current_pos[1] - env_test.HALF_WINDOW_SIZE:current_pos[1] + env_test.HALF_WINDOW_SIZE + 1]
        current_cropped_plan = current_cropped_plan.flatten().reshape(1, -1)
        current_pos = np.array(current_pos)
        torch_current_obs = torch.from_numpy(current_obs).float().unsqueeze(0).to(device)
        torch_current_pos = torch.from_numpy(current_pos).float().unsqueeze(0).to(device)
        torch_current_pos = torch_current_pos.unsqueeze(0)

        action, hidden_state_next, cell_state_next = test_agent.choose_action(current_obs, current_cropped_plan, hidden_state, cell_state)
        state_next, r, done = env_test.step(action)

        torch_action = torch.from_numpy(np.array(action).reshape(1,1,1)).float().to(device)
        next_obs = state_next[0]
        next_pos = state_next[2]
        torch_next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)

        predicted_pos, hidden_next, cell_next =Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
        predict_pos = predicted_pos[0].squeeze().detach().cpu().numpy()
        predict_next_pos = [0,0]
        predict_next_pos[0] = round(predict_pos[0])  
        predict_next_pos[1] = round(predict_pos[1]) 
        predict_next_pos = env_test.clip_position(predict_next_pos)

        distance_error_perstep = (predict_next_pos[0]-next_pos[0])**2 + (predict_next_pos[1]-next_pos[1])**2
        distance_error += distance_error_perstep   
        number_step += 1     

        reward_test += r
        if done:
            break
        state = state_next
        current_pos = predict_next_pos
        hidden_batch, cell_batch = hidden_next, cell_next
        hidden_state, cell_state = hidden_state_next, cell_state_next
    distance_error /= number_step
    distance_error_history.append(distance_error) 
    iou_test=iou(env_test.environment_memory,env_test.plan,env_test.HALF_WINDOW_SIZE,env_test.plan_height,env_test.plan_width)
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
statistic_results = pearsonr(distance_error_history, iou_history)
print(f"corrlation:",statistic_results[0], "and confidence:",statistic_results[1])

regr = linear_model.LinearRegression()
regr.fit(np.array(distance_error_history).reshape(-1, 1), np.array(iou_history))
print(regr.coef_)
print(regr.intercept_)

fig = plt.figure(figsize=(5, 5))
axe = fig.add_subplot(1, 1, 1)
axe.set_xlabel('Average position estimation error')
axe.set_xlim(0, 5)
axe.set_ylabel('IoU')
axe.scatter(distance_error_history, iou_history)
X = np.arange(0,5,0.1)
Y = regr.coef_[0]*X+regr.intercept_
axe.plot(X,Y.reshape(-1),"r")
plt.savefig('./curve.png')
# iou_test_total = iou_test_total / N_iteration_test
# secs = int(time.time() - start_time_test)
# mins = secs / 60
# secs = secs % 60
# env_test.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env, \
#                 best_iou=best_iou,best_step=best_step,best_brick=best_brick,best_plan=best_plan)


# info_file = save_path+"Seed"+str(seeds)+"_lr"+str(Lr)
# plt.savefig(info_file + '.png')
# with open(info_file + ".pkl",'wb') as f:
#     pickle.dump(iou_history, f)
# print('iou_all_min:',iou_min)
# print("iou_ave:",iou_test_total)
# STD = statistics.stdev(iou_history)
# print("std:",STD)
# print("best IOU:",best_iou)
# print("best_plan_index:",best_plan_index)
