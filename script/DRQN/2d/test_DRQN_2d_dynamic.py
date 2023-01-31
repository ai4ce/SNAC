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
from DMP_Env_2D_dynamic_usedata_plan import deep_mobile_printing_2d1r
import argparse

def main(args):
    def set_seed(seeds):
        torch.manual_seed(seeds)
        torch.cuda.manual_seed_all(seeds)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seeds)
        random.seed(seeds)
        os.environ['PYTHONHASHSEED'] = str(seeds)
    plans = ['dense', 'sparse']
    save_path = "./log/DRQN/plot/2D/Dynamic/"
    load_path = "./log/dynamic/DRQN_2d_"
    plan_choose = args.plan_choose
    PLAN_NAME = plans[plan_choose]
    seed = args.seed
    set_seed(seed)
    pth_plan = args.pth_plan
    if os.path.exists(save_path) == False:
        os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('2D_Static')
    iou_all_average = 0
    iou_all_min = 1
    env = deep_mobile_printing_2d1r(data_path="../../../Env/2D/data_2d_dynamic_"+PLAN_NAME+"_envplan_500_test.pkl", random_choose_paln=False)
    ######################
    # hyper parameter
    minibatch_size=64
    Lr=args.lr
    N_iteration=10000
    N_iteration_test=500
    alpha=0.9
    Replay_memory_size=1000
    Update_traget_period=200
    Action_dim=env.action_dim
    State_dim=env.state_dim
    hidden_state_dim=256
    Time_step=20
    UPDATE_FREQ=5
    INITIAL_EPSILON = 0.2
    FINAL_EPSILON = 0.0
    ######################
    print("State_dim:",State_dim)
    print("total_step:",env.total_step)
    print("Action_dim",Action_dim)

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
       
    test_agent=DQN_AGNET(device)
    log_path = load_path + PLAN_NAME + "_lr" + str(Lr) + "_seed_" + str(seed) + "/Eval_net_episode_"
    test_agent.Eval_net.load_state_dict(torch.load(log_path + str(pth_plan) +".pth" ,map_location='cuda:0'))
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
    iou_all_average += iou_test_total
    iou_all_min = min(iou_min,iou_all_min)
    total_reward=0
    state=env.reset()
    env_plan=env.input_plan
    print("total_brick:", env.total_brick)
    print('iou_his:',iou_history)
    print('iou_min:',min(iou_history))
    # env.render(ax)
    hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)
    env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)
    save_file = save_path + "Plan" + str(plan_choose) +'_seed'+str(seed)
    plt.savefig(save_file + '.png')
    with open(save_file + ".pkl",'wb') as f:
        pickle.dump(iou_history, f)
    iou_all_average = iou_all_average/20
    print('#### Finish #####')
    print('iou_all_average',iou_all_average)
    print('iou_all_min',iou_all_min)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-06, type=float, help='learning rate')
    parser.add_argument('--plan-choose', default=0, type=int, help='0 for dense, 1 for sparse')
    parser.add_argument('--seed', default=1, type=int, help='Random seed between 1 through 4')
    parser.add_argument('--pth-plan', default=1, type=int, help='Model number')
    
    args = parser.parse_args()
    print(args)
    main(args)