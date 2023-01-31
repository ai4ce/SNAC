import torch
import torch.nn as nn
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import time
import os
from collections import deque
import sys
import os
sys.path.append('../../../Env/1D/')
from DMP_Env_1D_static_Lnet_test import deep_mobile_printing_1d1r # whats the diff between normal and test?
import model

pth_plan = ['log/static_with_Lnet/DQN_1d_Lnet_sin_lr1e-05_seed_1/Eval_net_episode_2969.pth',
 'log/static_with_Lnet/DQN_1d_Lnet_gaussian_lr1e-05_seed_1/Eval_net_episode_2870.pth', 
 'log/static_with_Lnet/DQN_1d_Lnet_step_lr1e-05_seed_1/Eval_net_episode_1700.pth']

lnet_type = ["./1d/model_1D_explicit_sin_lstm/Lr_0.0001_batchsize_10/model_01000000.pth",
"./1d/model_1D_explicit_gaussian_lstm/Lr_0.0001_batchsize_10/model_01000000.pth",
"./1d/model_1D_explicit_step_lstm/Lr_0.0001_batchsize_10/model_01000000.pth"]

save_path = "./log/plot/1D/static_with_Lnet/lr1e-05/"

if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)

print('1D_Static')

iou_all_average = 0
iou_all_min = 1
for plan_choose in range(3):

    env = deep_mobile_printing_1d1r(plan_choose=plan_choose)
    ######################
    # hyper parameter
    minibatch_size = 2000
    Lr = 0.00001
    N_iteration = 3000
    #N_iteration_test = 10
    N_iteration_test = 500
    alpha = 0.9
    Replay_memory_size = 50000
    Update_traget_period = 200
    Action_dim = env.action_dim
    State_dim = env.state_dim
    UPDATE_FREQ = 1
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    seeds=4
    set_seed(seeds)
    ######################
    PLAN_DIM = env.plan_width
    env.reset()
    print("State_dimension:", State_dim)
    print("Total Brick: ", env.total_brick)
    print("plan_width", PLAN_DIM)
    print('######################')


    def get_and_init_FC_layer(din, dout):
        li = nn.Linear(din, dout)
        nn.init.xavier_uniform_(
        li.weight.data, gain=nn.init.calculate_gain('relu'))
        li.bias.data.fill_(0.)
        return li

    class Q_NET(nn.Module):
        def __init__(self, ):
            super(Q_NET, self).__init__()
            #self.fc_1 = get_and_init_FC_layer(State_dim+1, 64) # should this just be 6? obs = 5, pos = 1
            self.fc_1 = get_and_init_FC_layer(7, 64) # should this just be 7? obs = 5, pos = 1, act = 1
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
            self.Eval_net = Q_NET().to(device)
            self.Target_net = Q_NET().to(device)
            self.learn_step = 0                                     # counting the number of learning for update traget periodiclly 
            self.count_memory = 0                                         # counting the transitions 
            # self.replay_memory = np.zeros((Replay_memory_size, State_dim * 2 + 2)) 
            self.replay_memory = deque(maxlen=Replay_memory_size)    
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
            self.greedy_epsilon=0.2
            self.loss = nn.SmoothL1Loss()
            self.loss_his=[]
        def store_memory(self,s,a,r,s_next,pose_current, pose_next):
            self.replay_memory.append((s,a,r,s_next,pose_current, pose_next))
            self.count_memory+=1
        def choose_action(self, s, pose):      
            state = torch.FloatTensor(s).unsqueeze(0).to(device)
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
        def learning_process(self):
            self.optimizer.zero_grad()
            self.Eval_net.train()
            if self.learn_step% Update_traget_period == 0:
                self.Target_net.load_state_dict(self.Eval_net.state_dict())        
            minibatch = random.sample(self.replay_memory,minibatch_size)
            current_states = []
            acts = []
            rewards = []
            next_states = []
            current_pos = []
            predict_next_pos = []
            #state pose a
            for b in minibatch:
                current_states.append(b[0])
                acts.append(b[1])
                rewards.append(b[2])
                next_states.append(b[3])   
                current_pos.append(b[4])
                predict_next_pos.append(b[5])
            current_states = np.array(current_states)
            acts = np.array(acts).reshape(minibatch_size,1)
            rewards = np.array(rewards).reshape(minibatch_size,1)
            next_states = np.array(next_states)
            #current_pos = np.array(float(current_pos)) # sometimes, it's [2000], others it's [2000,1]
            current_pos = np.array(current_pos).astype(float).reshape(minibatch_size,1)
            predict_next_pos = np.array(predict_next_pos).astype(float)
            b_state = torch.from_numpy(current_states).float().squeeze(1).to(device)
            b_action = torch.from_numpy(acts).float().to(device)
            b_reward = torch.from_numpy(rewards).float().to(device)
            b_state_next = torch.from_numpy(next_states).float().squeeze(1).to(device)
            b_current_pos = torch.from_numpy(current_pos).float().to(device)
            b_predict_next_pos = torch.from_numpy(predict_next_pos).float().unsqueeze(-1).to(device)
            Q_eval = self.Eval_net(b_state,b_current_pos, b_action)
            Q=torch.zeros(minibatch_size,Action_dim).to(device)
            for item in range(Action_dim):
                a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                a=a.expand(minibatch_size,-1)
                #Q_value=self.Target_net(b_state_next,b_current_pos, a).detach().squeeze(1)
                Q_value=self.Target_net(b_state_next,b_predict_next_pos, a).detach().squeeze(1)
                Q[:,item]=Q_value 
            Q_next=torch.max(Q,dim=1)[0].reshape(minibatch_size,1)
            Q_target = b_reward + alpha * Q_next
            loss = self.loss(Q_eval, Q_target)
            loss.backward()
            self.optimizer.step()
            self.learn_step+=1
            train_loss=loss.item()
            return train_loss

    Lnet = model.SNAC_Lnet(input_size=12, hidden_size=128, device=device, Loss_type="L2")
    model_checkpoint = torch.load(lnet_type[plan_choose], map_location=device)
    Lnet.load_state_dict(model_checkpoint)
    Lnet.to(device)
    Lnet.eval()
    test_agent = DQN_AGNET(device)
    test_agent.Eval_net.load_state_dict(torch.load(pth_plan[plan_choose], map_location='cpu'))
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
        prev_state = env.reset()
        reward_test = 0
        hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
        prev_state = env.reset()
        obs = np.reshape(prev_state, (1,1,8))
        current_pos = obs[0][0][7:] #[0]

        while True:
            current_obs = obs[0][0][0:5]
            action = test_agent.choose_action(current_obs, current_pos)
            # state_next, r, done = env.step(action)
            new_obs,reward,done = env.step(action)
            torch_current_obs = torch.from_numpy(current_obs).view(1,1,5).float().to(device)
            torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
            torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
            new_obs = np.reshape(new_obs, (1,1,8))
            next_obs = new_obs[0][0][0:5]
            torch_next_obs = torch.from_numpy(next_obs).view(1,1,5).float().to(device)

            input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)
            predicted_pos, hidden_next, cell_next = Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
            predict_pos = predicted_pos[0][0][0].detach().cpu().numpy()
            predict_next_pos = round(predict_pos[0])

            reward_test += reward
            if done:
                break
            obs = new_obs
            current_pos = np.array([float(predict_next_pos)])
            hidden_batch, cell_batch = hidden_next, cell_next

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
    plt.savefig(save_path+"Plan"+str(plan_choose)+"_Seed"+str(seeds)+'.png')

iou_all_average = iou_all_average/3
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()