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
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_dynamic_test import deep_mobile_printing_2d1r
import statistics
def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)
## hyper parameter
seeds=5
set_seed(seeds)
Lr= 0.00001
pth_plan = ["7639"]
Replay_memory_size=50000
PALN_CHOICE=1  # 0 dense, 1 sparse
PLAN_LIST=["dense","sparse"]
PLAN_NAME=PLAN_LIST[PALN_CHOICE]
OUT_FILE_NAME="DRQN_hindsight_2d_"+PLAN_NAME+"_lr"+str(Lr)+"_seed_"+str(seeds)
print(OUT_FILE_NAME)
save_path = "mnt/NAS/home/WenyuHan/SNAC/DRQN_Hindsight/2D/plot/"
load_path = "/mnt/NAS/home/WenyuHan/SNAC/DRQN_Hindsight/2D/log/dynamic/"+OUT_FILE_NAME+"/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print('2D_dynamic')
iou_all_average = 0
iou_all_min = 1
std_all = 0
for test_set in range(10):
    env = deep_mobile_printing_2d1r(plan_choose=PALN_CHOICE, test_set=test_set)
    N_iteration_test=200
    Replay_memory_size=1000
    Action_dim=env.action_dim
    State_dim=env.state_dim
    hidden_state_dim=256
    Time_step=20
    UPDATE_FREQ=5

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
        def __init__(self,out_size,hidden_size):
            super(Q_NET, self).__init__()

            self.out_size = out_size
            self.hidden_size = hidden_size

            self.conv_layer1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=2) # potential check - in_channels
            self.conv_layer2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2)
            self.conv_layer3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2)
            self.conv_layer4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2)



            self.fc_1 = get_and_init_FC_layer(State_dim, 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.rnn=nn.LSTM(128+32,hidden_size,num_layers=1,batch_first=True)
            self.adv  = get_and_init_FC_layer(hidden_size, self.out_size)
            self.val  = get_and_init_FC_layer(hidden_size, 1)
            self.relu = nn.ReLU()
        def forward(self,x,plan,bsize,time_step,hidden_state,cell_state):

            plan = plan.view(bsize*time_step,1,20,20)

            conv_out = self.conv_layer1(plan)
            conv_out = self.relu(conv_out)
            conv_out = self.conv_layer2(conv_out)
            conv_out = self.relu(conv_out)

            conv_out = self.conv_layer3(conv_out)
            conv_out = self.relu(conv_out)


            conv_out = conv_out.view(bsize*time_step,32)

            x=x.view(bsize*time_step,State_dim)


            x = self.fc_1(x)
            x = self.relu(x)
            x = self.fc_2(x)
            x = self.relu(x)
            x = self.fc_3(x)
            x = self.relu(x)
            x=torch.cat((x,conv_out),dim=1)
            x = x.view(bsize,time_step,128+32)
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



    class Memory():

        def __init__(self,memsize):
            self.memsize = memsize
            self.memory = deque(maxlen=self.memsize)

        def add_episode(self,epsiode):
            self.memory.append(epsiode)

        def get_batch(self,bsize,time_step):
            sampled_epsiodes = random.sample(self.memory,bsize)
            batch = []
            for episode in sampled_epsiodes:
                point = np.random.randint(0,len(episode)+1-time_step)
                batch.append(episode[point:point+time_step])
            return batch


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
            self.greedy_epsilon=0.2
            self.replaymemory=Memory(Replay_memory_size)
        def choose_action(self,s,plan,hidden_state,cell_state):

            state=torch.from_numpy(s).float().to(self.device)
            env_plan=torch.from_numpy(plan).float().to(self.device)
            model_out = self.Eval_net.forward(state,env_plan,bsize=1,time_step=1,hidden_state=hidden_state,cell_state=cell_state)
            out = model_out[0]
            action = int(torch.argmax(out[0]))
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]

            return action, hidden_state, cell_state
        def learning_process(self):

            self.optimizer.zero_grad()
            self.Eval_net.train()
            if self.learn_step% Update_traget_period == 0:
                self.Target_net.load_state_dict(self.Eval_net.state_dict())

            hidden_batch, cell_batch = self.Eval_net.init_hidden_states(bsize=minibatch_size)
            batch = self.replaymemory.get_batch(bsize=minibatch_size,time_step=Time_step)

            current_states = []
            acts = []
            rewards = []
            next_states = []
            env_plans = []
            for b in batch:
                cs,ac,rw,ns,ep = [],[],[],[],[]
                for element in b:
                    cs.append(element[0])
                    ac.append(element[1])
                    rw.append(element[2])
                    ns.append(element[3])
                    ep.append(element[4])
                current_states.append(cs)
                acts.append(ac)
                rewards.append(rw)
                next_states.append(ns)
                env_plans.append(ep)
            current_states = np.array(current_states)
            acts = np.array(acts)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            env_plans = np.array(env_plans)

            torch_current_states = torch.from_numpy(current_states).float().to(self.device)
            torch_acts = torch.from_numpy(acts).long().to(self.device)
            torch_rewards = torch.from_numpy(rewards).float().to(self.device)
            torch_next_states = torch.from_numpy(next_states).float().to(self.device)
            torch_env_plans = torch.from_numpy(env_plans).float().to(self.device)


            Q_s, _ = self.Eval_net.forward(torch_current_states,torch_env_plans,bsize=minibatch_size,time_step=Time_step,hidden_state=hidden_batch,cell_state=cell_batch)

            Q_s_a = Q_s.gather(dim=1,index=torch_acts[:,Time_step-1].unsqueeze(dim=1)).squeeze(dim=1)


            Q_next,_ = self.Target_net.forward(torch_next_states,torch_env_plans,bsize=minibatch_size,time_step=Time_step,hidden_state=hidden_batch,cell_state=cell_batch)
            Q_next_max,__ = Q_next.detach().max(dim=1)
            target_values = torch_rewards[:,Time_step-1] + (alpha * Q_next_max)



            loss = self.loss(Q_s_a, target_values)

            loss.backward()
            self.optimizer.step()
            self.learn_step+=1
            self.loss_his.append(loss.item())


    test_agent=DQN_AGNET(device)
    log_path = load_path  + "Eval_net_episode_"
    test_agent.Eval_net.load_state_dict(torch.load(log_path + pth_plan[0]+".pth" ,map_location=device))
    best_iou=0
    best_env = np.array([])
    iou_test_total=0
    reward_test_total=0
    iou_min = 1
    iou_history = []
    start_time_test = time.time()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)

    for i in range(N_iteration_test):
        state = env.reset()
        prev_state=state[0]
        env_plan=state[1]
        reward_test=0
        print('Ep:',i)

        hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)

        while True:
            action,hidden_state_next, cell_state_next = test_agent.choose_action(prev_state,env_plan,hidden_state, cell_state)
            state_next, r, done = env.step(action)
            reward_test += r
            if done:

                break
            prev_state = state_next[0]
            hidden_state, cell_state = hidden_state_next, cell_state_next
        iou_test=iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
        iou_history.append(iou_test)

        iou_min = min(iou_min,iou_test)
        if iou_test > best_iou:
            best_iou = iou_test
            best_step = env.count_step
            best_brick = env.count_brick
            best_plan = env.plan
            best_tb = env.total_brick
            best_env = env.environment_memory

        iou_test_total+=iou_test
        reward_test_total+=reward_test

    reward_test_total=reward_test_total/N_iteration_test
    iou_test_total=iou_test_total/N_iteration_test
    secs = int(time.time() - start_time_test)
    mins = secs / 60
    secs = secs % 60
    STD = statistics.stdev(iou_history)

    iou_all_average += iou_test_total
    iou_all_min = min(iou_min,iou_all_min)

    # total_reward=0
    # state=env.reset()
    # env_plan=env.input_plan

    # print("total_brick:", env.total_brick)
    # print('iou_his:',iou_history)
    # print('iou_min:',min(iou_history))
    # # env.render(ax)
    # hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)
    # env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)
    # plt.savefig(save_path + "Plan" + str(plan_choose) +'_test'+str(test_set) + '.png')

    # while True:
    #     action,hidden_state_next, cell_state_next = test_agent.choose_action(state,env_plan,hidden_state, cell_state)
    #     state_next, r, done = env.step(action)
    #     # env.render(ax,iou_min,iou_test_total,N_iteration_test)
    #     # plt.pause(0.1)
    #     total_reward+=r
    #     if done:
    #         env.render(ax, iou_min, iou_test_total, N_iteration_test)
    #         plt.savefig(save_path + "Plan" + str(plan_choose) +'_test'+str(test_set) + '.png')
    #         break
    #     state = state_next
    #     hidden_state, cell_state = hidden_state_next, cell_state_next
std_all += STD
std_all = std_all/10
iou_all_average = iou_all_average/10
print('#### Finish #####')
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
print("std:",std_all)
