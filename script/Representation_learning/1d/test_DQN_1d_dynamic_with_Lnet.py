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
from DMP_Env_1D_dynamic_usedata_plan import deep_mobile_printing_1d1r # whats the diff between normal and test?
import model

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)

pth_plan = '1449'
save_path = "./log/plot/1D/Dynamic/"
load_path = "./log/dynamic/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lnet_file = "./1d/model_1D_explicit_dynamic_lstm/Lr_0.0001_batchsize_10/model_01000000.pth"

print('1D_Dynamic')
print("device_using:", device)
iou_all_average = 0
iou_all_min = 1
for seed in range(1,5):
    set_seed(seed)
    env = deep_mobile_printing_1d1r(data_path="../../../Env/1D/data_1d_dynamic_sin_envplan_500_test.pkl", random_choose_paln=False)
    ######################
    # hyper parameter
    minibatch_size = 2000
    Lr = 0.001
    N_iteration = 3000
    N_iteration_test = 500
    alpha = 0.9
    Replay_memory_size = 50000
    Update_traget_period = 200
    UPDATE_FREQ = 1
    Action_dim = env.action_dim
    State_dim = env.state_dim
    plan_dim = env.plan_width
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
        nn.init.xavier_uniform_(
        li.weight.data, gain=nn.init.calculate_gain('relu'))
        li.bias.data.fill_(0.)
        return li

    class Q_NET(nn.Module):
        def __init__(self, ):
            super(Q_NET, self).__init__()
            self.fc_1 = get_and_init_FC_layer(7+plan_dim, 64) # should this just be 7? obs = 5, pos = 1, act = 1
            # we need to modify this input to for dynamic which is sars,env_plan and curr_pos
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.out = get_and_init_FC_layer(128, 1)
            self.relu = nn.ReLU()
        def forward(self, s, pose, a, plan):
            x=torch.cat((s,pose,a,plan),dim=1)
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
        def store_memory(self,s,a,r,s_next,pose_current,pose_next,env_plan):
            self.replay_memory.append((s,a,r,s_next,pose_current, pose_next,env_plan))
            self.count_memory+=1
        def choose_action(self, s, pose, env_plan):      
            state = torch.FloatTensor(s).unsqueeze(0).to(device)
            pose = torch.FloatTensor(pose).unsqueeze(0).to(device)
            env_plan=torch.from_numpy(env_plan).unsqueeze(0).float().to(device)
            choose=np.random.uniform()
            Q=[]
            if choose<=self.greedy_epsilon:
                action=np.random.randint(0, Action_dim)         
            else:
                for item in range(Action_dim):
                    a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                    Q_value=self.Eval_net(state, pose, a, env_plan)
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
            env_plans = []
            #state pose a
            for b in minibatch:
                current_states.append(b[0])
                acts.append(b[1])
                rewards.append(b[2])
                next_states.append(b[3])   
                current_pos.append(b[4])
                predict_next_pos.append(b[5])
                env_plans.append(b[6])
            current_states = np.array(current_states)
            acts = np.array(acts).reshape(minibatch_size,1)
            rewards = np.array(rewards).reshape(minibatch_size,1)
            next_states = np.array(next_states)
            current_pos = np.array(current_pos).astype(float).reshape(minibatch_size,1)
            predict_next_pos = np.array(predict_next_pos).astype(float)
            env_plans = np.array(env_plans).astype(float)
            b_state = torch.from_numpy(current_states).float().squeeze(1).to(device)
            b_action = torch.from_numpy(acts).float().to(device)
            b_reward = torch.from_numpy(rewards).float().to(device)
            b_state_next = torch.from_numpy(next_states).float().squeeze(1).to(device)
            b_current_pos = torch.from_numpy(current_pos).float().to(device)
            b_predict_next_pos = torch.from_numpy(predict_next_pos).float().unsqueeze(-1).to(device)
            b_env_plans = torch.from_numpy(env_plans).float().to(device)
            Q_eval = self.Eval_net(b_state,b_current_pos, b_action, b_env_plans)
            Q=torch.zeros(minibatch_size,Action_dim).to(device)
            for item in range(Action_dim):
                a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                a=a.expand(minibatch_size,-1)
                Q_value=self.Target_net(b_state_next,b_predict_next_pos,a,b_env_plans).detach().squeeze(1)
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
    model_checkpoint = torch.load(lnet_file, map_location=device) # MAKE SURE THIS IS PROPER FILE
    Lnet.load_state_dict(model_checkpoint)
    Lnet.to(device)
    Lnet.eval()

    test_agent = DQN_AGNET(device)
    log_path = load_path + "DQN_1d_sin_lr5e-05_seed_5/Eval_net_episode_"

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
        hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
        reward_test = 0
        prev_state = env.reset()
        obs = np.reshape(prev_state[0], (1,1,8))
        env_plan = prev_state[1]
        current_pos = obs[0][0][7:]

        while True:
            current_obs = obs[0][0][0:5]
            # current_pos = np.array([float(current_pos)])
            action = test_agent.choose_action(current_obs, current_pos, env_plan)
            new_obs,reward,done = env.step(action)
            
            torch_current_obs = torch.from_numpy(current_obs).view(1,1,5).float().to(device)
            torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
            torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
            new_obs = np.reshape(new_obs[0], (1,1,8))
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
            best_step = env.count_step
            best_brick = env.conut_brick
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
               best_iou=best_iou, best_step=best_step, best_brick=best_brick)

    iou_all_average += iou_test_total
    iou_all_min = min(iou_min,iou_all_min)
    plt.savefig(save_path+"Plan_one_hot_"+str(seed)+'.png')

iou_all_average = iou_all_average/10
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()
