import torch
import torch.nn as nn
import numpy as np
import sys
import pickle
import random
import matplotlib.pyplot as plt
import time
import os
from collections import deque
sys.path.append('../../../Env/1D/')
from DMP_Env_1D_dynamic_usedata_plan import deep_mobile_printing_1d1r
import model
import argparse

def main(args):
    lr = args.lr
    pth_plan = args.pth_plan
    seed = args.seed
    save_path = "./log/plot/1D_Dynamic_DRQN_with_Lnet/"
    load_path = "./log/dynamic_with_Lnet/DRQN_1d_Lnet_dynamic_lr" + str(lr) + "_seed_" + str(seed)

    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('1D_Dynamic')
    print("device_using:", device)
    iou_all_average = 0
    iou_all_min = 1

    env = deep_mobile_printing_1d1r(data_path="../../../Env/1D/data_1d_dynamic_sin_envplan_500_test.pkl")
    ######################
    # hyper parameter
    minibatch_size = 64
    N_iteration = 3000
    N_iteration_test = 500
    alpha = 0.9
    Replay_memory_size = 1000
    Update_traget_period = 200
    UPDATE_FREQ = 1
    Action_dim = env.action_dim
    State_dim = 6
    plan_dim = env.plan_width
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    hidden_state_dim = 256
    Time_step = 20
    ######################
    env.reset()
    print("State_dimension:", State_dim)
    print("Total Brick: ", env.total_brick)
    print("plan_width", plan_dim)
    print('######################')


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
            self.fc_1 = get_and_init_FC_layer(State_dim+plan_dim, 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.rnn=nn.LSTM(128,hidden_size,num_layers=1,batch_first=True)
            self.adv  = get_and_init_FC_layer(hidden_size, self.out_size)
            self.val  = get_and_init_FC_layer(hidden_size, 1)
            self.relu = nn.ReLU()

        def forward(self,x,plan,bsize,time_step,hidden_state,cell_state):
            x=x.view(bsize * time_step, State_dim) 
            plan = plan.view(bsize * time_step, plan_dim)
            x = torch.cat((x, plan), dim=1)
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

    class DRQN_AGNET():
        def __init__(self,device):
            self.Eval_net = Q_NET(Action_dim, hidden_size=hidden_state_dim).to(device)
            self.Target_net = Q_NET(Action_dim, hidden_size=hidden_state_dim).to(device)
            self.learn_step = 0                                     # counting the number of learning for update traget periodiclly 
            self.replay_memory = Memory(Replay_memory_size)    
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=lr)
            self.greedy_epsilon=0.2
            self.loss = nn.SmoothL1Loss()
            self.loss_his=[]
            self.device = device

        def choose_action(self, s, pose, plan, hidden_state, cell_state):      
            state = torch.FloatTensor(s).to(self.device)
            pose = torch.FloatTensor(pose).to(self.device)
            env_plan = torch.from_numpy(plan).float().to(self.device)
            choose=np.random.uniform()
            Q=[]
            if choose<=self.greedy_epsilon:
                x=torch.cat((state,pose),dim=0)
                model_out = self.Eval_net.forward(x, env_plan, bsize=1, time_step=1, hidden_state=hidden_state,
                        cell_state=cell_state)
                action=np.random.randint(0, Action_dim)
                hidden_state = model_out[1][0]
                cell_state = model_out[1][1]
            else:
                x=torch.cat((state,pose),dim=0)
                model_out = self.Eval_net.forward(x, env_plan, bsize=1, time_step=1, hidden_state=hidden_state,
                        cell_state=cell_state)
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
            minibatch = self.replay_memory.get_batch(bsize=minibatch_size,time_step=Time_step)  
            current_states = []
            current_pose = []
            acts = []
            rewards = []
            next_states = []
            next_pose = []
            env_plans = []
            for b in minibatch:
                cs,cp,ac,rw,ns,nps,ep = [],[],[],[],[],[],[]
                for element in b:
                    cs.append(element[0])
                    ac.append(element[1])
                    rw.append(element[2])
                    ns.append(element[3])
                    cp.append(element[4])
                    nps.append(element[5])
                    ep.append(element[6])
                current_states.append(cs)
                acts.append(ac)
                rewards.append(rw)
                next_states.append(ns)
                current_pose.append(cp)
                next_pose.append(nps)
                env_plans.append(ep)
            current_states = np.array(current_states)
            current_pose = np.array(current_pose)
            acts = np.array(acts)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            next_pose = np.array(next_pose)
            env_plans = np.array(env_plans)

            b_state = torch.from_numpy(current_states).view(minibatch_size, Time_step, -1).float().to(device)
            b_current_pose = torch.from_numpy(current_pose.astype(float)).view(minibatch_size, Time_step, -1).float().to(device)
            b_action = torch.from_numpy(acts).long().to(device)
            b_reward = torch.from_numpy(rewards).float().to(device)
            b_state_next = torch.from_numpy(next_states).view(minibatch_size, Time_step, -1).float().to(device)
            b_next_pose = torch.from_numpy(next_pose.astype(float)).view(minibatch_size, Time_step, -1).float().to(device)
            b_env_plans = torch.from_numpy(env_plans).view(minibatch_size, Time_step, -1).float().to(self.device)

            eval_net_input = torch.cat((b_state, b_current_pose), dim=2)
            Q_s, _ = self.Eval_net.forward(eval_net_input,b_env_plans,bsize=minibatch_size,time_step=Time_step,hidden_state=hidden_batch,cell_state=cell_batch)
            Q_s_a = Q_s.gather(dim=1,index=b_action[:,Time_step-1].unsqueeze(dim=1)).squeeze(dim=1)

            target_net_input = torch.cat((b_state_next, b_next_pose), dim=2)
            Q_next,_ = self.Target_net.forward(target_net_input,b_env_plans,bsize=minibatch_size,time_step=Time_step,hidden_state=hidden_batch,cell_state=cell_batch)
            Q_next_max,__ = Q_next.detach().max(dim=1)
            target_values = b_reward[:,Time_step-1] + (alpha * Q_next_max)
            loss = self.loss(Q_s_a, target_values)
            loss.backward()
            self.optimizer.step()
            self.learn_step+=1
            train_loss=loss.item()
            return train_loss

    Lnet = model.SNAC_Lnet(input_size=12, hidden_size=128, device=device, Loss_type="L2")
    model_checkpoint = torch.load("./1d/model_1D_explicit_dynamic_lstm/Lr_0.0001_batchsize_10/model_01000000.pth", map_location=device)
    Lnet.load_state_dict(model_checkpoint)
    Lnet.to(device)
    Lnet.eval()

    test_agent = DRQN_AGNET(device)
    log_path = load_path + "/Eval_net_episode_"
    test_agent.Eval_net.load_state_dict(torch.load(log_path + pth_plan + ".pth", map_location='cpu'))

    test_agent.greedy_epsilon = 0.08
    best_iou = 0
    best_env = np.array([])
    iou_test_total = 0
    iou_history = []
    iou_min = 1
    reward_test_total = 0
    start_time_test = time.time()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    for ep in range(N_iteration_test):
        prev_state = env.reset()
        obs = np.reshape(prev_state[0], (1,1,8))
        current_pos = obs[0][0][7:] #[0]
        env_plan = prev_state[1]
        hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
        hidden_state, cell_state = test_agent.Eval_net.init_hidden_states(bsize=1)
        reward_test=0
        while True:
            current_obs = obs[0][0][0:5]

            action, hidden_state_next, cell_state_next = test_agent.choose_action(current_obs, current_pos, env_plan, hidden_state, cell_state)
            new_obs,reward,done = env.step(action)
            
            torch_current_obs = torch.from_numpy(current_obs).view(1,1,5).float().to(device)
            torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
            torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
            # new_obs = np.reshape(new_obs, (1,1,8))
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
            hidden_state, cell_state = hidden_state_next, cell_state_next

        iou_test = env.iou()
        iou_history.append(iou_test)
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
    info_file = save_path+"Seed"+str(seed)+"_lr"+str(lr)
    plt.savefig(info_file + '.png')
    with open(info_file + ".pkl",'wb') as f:
        pickle.dump(iou_history, f)
    iou_all_average = iou_all_average/10
    print('iou_all_average',iou_all_average)
    print('iou_all_min',iou_all_min)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-06, type=float, help='learning rate')
    parser.add_argument('--pth-plan', default="1327", type=str, help='model number')
    parser.add_argument('--seed', default=1, type=int, help='Random seed between 1 through 4')
    
    args = parser.parse_args()
    print(args)
    main(args)