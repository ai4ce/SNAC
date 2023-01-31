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
import sys
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_dynamic_usedata_plan import deep_mobile_printing_2d1r
import argparse

def main(args):
    lr = args.lr
    seed = args.seed
    pth_plan = args.pth_plan
    save_path = "./log/plot/2D_Dynamic/"
    load_path = "./log/dynamic/DQN_2d_"
    test_set = ['../../../Env/2D/data_2d_dynamic_dense_envplan_500_test.pkl',
        '../../../Env/2D/data_2d_dynamic_sparse_envplan_500_test.pkl']
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    plans = ['dense', 'sparse']
    plan_choose = args.plan_choose
    print('2D_Dynamic')
    iou_all_average = 0
    iou_all_min = 1
    env = deep_mobile_printing_2d1r(data_path=test_set[plan_choose], random_choose_paln=False)
    ######################
    # hyper parameter
    minibatch_size = 2000
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
        nn.init.xavier_uniform_(
        li.weight.data, gain=nn.init.calculate_gain('relu'))
        li.bias.data.fill_(0.)
        return li

    class Q_NET(nn.Module):
        def __init__(self, ):
            super(Q_NET, self).__init__()
            self.conv_layer1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=2) # potential check - in_channels
            self.conv_layer2 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2)
            self.conv_layer3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2)
            self.conv_layer4 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=2)
            self.fc_1 = get_and_init_FC_layer(State_dim+1+32, 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.out = get_and_init_FC_layer(128, 1)
            self.relu = nn.ReLU()
        def forward(self, s,a,plan,bsize):
            plan = plan.view(bsize,1,20,20) 
            conv_out = self.conv_layer1(plan)
            conv_out = self.relu(conv_out)
            conv_out = self.conv_layer2(conv_out)
            conv_out = self.relu(conv_out)
            conv_out = self.conv_layer3(conv_out)
            conv_out = self.relu(conv_out)
            conv_out = conv_out.view(bsize,32)
            x=torch.cat((s,a,conv_out),dim=1)
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
            self.replay_memory = deque(maxlen=Replay_memory_size)    
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=lr)
            self.greedy_epsilon=0.2
            self.loss = nn.SmoothL1Loss()
            self.loss_his=[]
        def store_memory(self,s,a,r,s_next,env_plan):
            self.replay_memory.append((s,a,r,s_next,env_plan))
            self.count_memory+=1
        def choose_action(self,s,env_plan):
            state=torch.FloatTensor(s).to(device)
            env_plan=torch.from_numpy(env_plan).float().to(device)
            env_plan=env_plan.unsqueeze(0)
            choose=np.random.uniform()
            Q=[]
            if choose<=self.greedy_epsilon:
                action=np.random.randint(0, Action_dim)         
            else:
                for item in range(Action_dim):
                    a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                    Q_value=self.Eval_net(state,a,env_plan,1)
                    Q.append(Q_value.item())  
                action=np.argmax(Q)    
            return action

    test_agent=DQN_AGNET(device)
    log_path = load_path + plans[plan_choose] + "_lr" + str(lr) + "_seed_" + str(seed) + "/Eval_net_episode_"
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
        while True:
            action = test_agent.choose_action(prev_state, env_plan)
            state_next, r, done = env.step(action)
            reward_test += r
            if done:
                break
            prev_state = state_next[0]
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
    env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)
    info_file = save_path + "Plan" + str(plan_choose) +'_lr'+ str(lr) + '_seed' + str(seed)
    plt.savefig(info_file + '.png')
    with open(info_file + ".pkl",'wb') as f:
        pickle.dump(iou_history, f)
    iou_all_average = iou_all_average/20
    print('#### Finish #####')
    print('iou_all_average',iou_all_average)
    print('iou_all_min',iou_all_min)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-06, type=float, help='learning rate')
    parser.add_argument('--pth-plan', default="1327", type=str, help='model number')
    parser.add_argument('--plan-choose', default=0, type=int, help='0 for dense, 1 for sparse')
    parser.add_argument('--seed', default=1, type=int, help='Random seed between 1 through 4')
    args = parser.parse_args()
    print(args)
    main(args)