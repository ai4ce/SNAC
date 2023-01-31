import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append('../../../Env/1D/')
sys.path.append('../../../')
import utils
from DMP_Env_1D_dynamic_usedata_plan import deep_mobile_printing_1d1r
import time
import os
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
import model
import argparse

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seeds)
    random.seed(seeds)
    os.environ['PYTHONHASHSEED'] = str(seeds)

def main(cfg):
    config_path = cfg.config_path
    if config_path is None:
        print('Please provide a config path')
        return
    args = utils.read_config(config_path)
    ## hyper parameter
    seeds=args.Random_seed
    set_seed(seeds)
    minibatch_size=args.batch_size
    Lr=args.lr
    N_iteration=args.N_iteration
    N_iteration_test=10
    alpha=0.9
    Replay_memory_size=args.Replay_buffer_size
    Update_traget_period=args.update_traget_period
    hidden_state_dim = 256
    Time_step = 20
    UPDATE_FREQ = 1
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    PALN_CHOICE=args.plan_type  ##0: dense 1: sparse
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    PLAN_LIST=["sin"]
    PLAN_NAME=PLAN_LIST[PALN_CHOICE]
    OUT_FILE_NAME="DRQN_1d_dynamic_Lnet_"+PLAN_NAME+"_lr"+str(Lr)+"_seed_"+str(seeds)
    print(OUT_FILE_NAME)
    model_dir=args.model_dir+"/"+OUT_FILE_NAME+"/"
    if os.path.exists(model_dir)==False:
        os.makedirs(model_dir)
    env_train = deep_mobile_printing_1d1r(data_path="../../../Env/1D/data_1d_dynamic_sin_envplan_500_train.pkl")
    env_val = deep_mobile_printing_1d1r(data_path="../../../Env/1D/data_1d_dynamic_sin_envplan_500_val.pkl",random_choose_paln=False)
    Action_dim=env_train.action_dim
    print(Action_dim)
    State_dim = env_train.state_dim
    plan_dim=env_train.plan_width
    print("state_dim",env_train.state_dim)
    print("total_step",env_train.total_step)
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
        def __init__(self, out_size, hidden_size):
            super(Q_NET, self).__init__()
            self.out_size = out_size
            self.hidden_size = hidden_size
            self.fc_1 = get_and_init_FC_layer(State_dim+State_dim-2, 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.rnn=nn.LSTM(128,hidden_size,num_layers=1,batch_first=True)
            self.adv  = get_and_init_FC_layer(hidden_size, self.out_size)
            self.val  = get_and_init_FC_layer(hidden_size, 1)
            self.relu = nn.ReLU()

        def forward(self,x,cropped_plan,bsize,time_step,hidden_state,cell_state):
            x=x.view(bsize * time_step, State_dim) 
            plan = cropped_plan.view(bsize * time_step, 5)
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
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
            self.greedy_epsilon=0.2
            self.loss = nn.SmoothL1Loss()
            self.loss_his=[]
            self.device = device

        def choose_action(self, s, croppedplan, hidden_state, cell_state):      
            state = torch.FloatTensor(s).to(self.device)
            env_plan = torch.from_numpy(croppedplan).float().to(self.device)
            choose=np.random.uniform()
            Q=[]
            if choose<=self.greedy_epsilon:
                model_out = self.Eval_net.forward(state, env_plan, bsize=1, time_step=1, hidden_state=hidden_state,
                        cell_state=cell_state)
                action=np.random.randint(0, Action_dim)
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
        def learning_process(self):
            self.optimizer.zero_grad()
            self.Eval_net.train()
            if self.learn_step% Update_traget_period == 0:
                self.Target_net.load_state_dict(self.Eval_net.state_dict())     
            hidden_batch, cell_batch = self.Eval_net.init_hidden_states(bsize=minibatch_size)
            minibatch = self.replay_memory.get_batch(bsize=minibatch_size,time_step=Time_step)  
            current_states = []
            current_copppedplan = []
            acts = []
            rewards = []
            next_states = []
            next_croppedplan = []
            for b in minibatch:
                cs,cp,ac,rw,ns,nps = [],[],[],[],[],[]
                for element in b:
                    cs.append(element[0])
                    ac.append(element[1])
                    rw.append(element[2])
                    ns.append(element[3])
                    cp.append(element[4])
                    nps.append(element[5])
                current_states.append(cs)
                acts.append(ac)
                rewards.append(rw)
                next_states.append(ns)
                current_copppedplan.append(cp)
                next_croppedplan.append(nps)
            current_states = np.array(current_states)
            current_copppedplan = np.array(current_copppedplan)
            acts = np.array(acts)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            next_croppedplan = np.array(next_croppedplan)
            b_state = torch.from_numpy(current_states).view(minibatch_size, Time_step, -1).float().to(device)
            b_current_croppedplan = torch.from_numpy(current_copppedplan).view(minibatch_size, Time_step, -1).float().to(device)
            b_action = torch.from_numpy(acts).long().to(device)
            b_reward = torch.from_numpy(rewards).float().to(device)
            b_state_next = torch.from_numpy(next_states).view(minibatch_size, Time_step, -1).float().to(device)
            b_next_croppedplan = torch.from_numpy(next_croppedplan).view(minibatch_size, Time_step, -1).float().to(device)
            Q_s, _ = self.Eval_net.forward(b_state,b_current_croppedplan,bsize=minibatch_size,time_step=Time_step,hidden_state=hidden_batch,cell_state=cell_batch)
            Q_s_a = Q_s.gather(dim=1,index=b_action[:,Time_step-1].unsqueeze(dim=1)).squeeze(dim=1)
            Q_next,_ = self.Target_net.forward(b_state_next,b_next_croppedplan,bsize=minibatch_size,time_step=Time_step,hidden_state=hidden_batch,cell_state=cell_batch)
            Q_next_max,__ = Q_next.detach().max(dim=1)
            target_values = b_reward[:,Time_step-1] + (alpha * Q_next_max)
            loss = self.loss(Q_s_a, target_values)
            loss.backward()
            self.optimizer.step()
            self.learn_step+=1
            train_loss=loss.item()
            return train_loss
    Lnet = model.SNAC_Lnet(input_size=16, hidden_size=128, device=device, Loss_type="L2")
    model_checkpoint = torch.load(args.pretrainmodel, map_location=device)
    Lnet.load_state_dict(model_checkpoint)
    Lnet.to(device)
    Lnet.eval()
    agent=DRQN_AGNET(device)
    number_steps=0
    total_steps = 0
    for i in range(0, Replay_memory_size):
        hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
        reward_train = 0
        prev_state = env_train.reset()
        env_total_brick = env_train.total_brick
        env_total_step = env_train.total_step
        current_pos = prev_state[3]
        start_time = time.time()
        local_memory = []
        while True:        
            total_steps +=1
            current_cropped_plan = env_train.plan_withborder[:,
                            current_pos - env_train.HALF_WINDOW_SIZE:current_pos + env_train.HALF_WINDOW_SIZE + 1]
            current_pos = np.array(current_pos)
            current_obs = prev_state[0]
            current_obs_normalized = prev_state[1]
            action = np.random.randint(0,Action_dim)
            new_state,reward,done = env_train.step(action)
            torch_current_obs = torch.from_numpy(current_obs).view(1,1,7).float().to(device)
            torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
            torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
            next_obs = new_state[0]
            next_obs_normalized = new_state[1]
            torch_next_obs = torch.from_numpy(next_obs).view(1,1,7).float().to(device)
            input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)
            predicted_pos, hidden_next, cell_next = Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
            predict_pos = predicted_pos[0][0][0].detach().cpu().numpy()
            predict_next_pos = round(predict_pos[0])
            predict_next_pos = env_train.clip_position(predict_next_pos)
            next_cropped_plan = env_train.plan_withborder[:,
                            predict_next_pos - env_train.HALF_WINDOW_SIZE:predict_next_pos + env_train.HALF_WINDOW_SIZE + 1]
            local_memory.append((current_obs_normalized, action, reward, next_obs_normalized, current_cropped_plan, next_cropped_plan))
            prev_state = new_state
            current_pos = predict_next_pos
            hidden_batch, cell_batch = hidden_next, cell_next
            number_steps+=1
            if done:
                break
        agent.replay_memory.add_episode(local_memory)
    agent.greedy_epsilon=INITIAL_EPSILON
    best_reward=0
    total_steps = 0
    writer = SummaryWriter(args.log_dir+"/"+OUT_FILE_NAME)
    for episode in range(N_iteration):
        hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
        hidden_state, cell_state = agent.Eval_net.init_hidden_states(bsize=1)
        prev_state = env_train.reset()
        current_pos = prev_state[3]
        env_total_brick = env_train.total_brick
        env_total_step = env_train.total_step
        reward_train = 0
        local_memory = []
        start_time = time.time()
        while True:
            total_steps +=1
            current_cropped_plan = env_train.plan_withborder[:,
                            current_pos - env_train.HALF_WINDOW_SIZE:current_pos + env_train.HALF_WINDOW_SIZE + 1]
            current_pos = np.array(current_pos)
            current_obs = prev_state[0]
            current_obs_normalized = prev_state[1]
            action, hidden_state_next, cell_state_next = agent.choose_action(current_obs_normalized, current_cropped_plan, hidden_state, cell_state)
            new_state, reward, done = env_train.step(action)
            torch_current_obs = torch.from_numpy(current_obs).view(1,1,7).float().to(device)
            torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
            torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
            next_obs = new_state[0]
            next_obs_normalized = new_state[1]
            torch_next_obs = torch.from_numpy(next_obs).view(1,1,7).float().to(device)
            input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)
            predicted_pos, hidden_next, cell_next = Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
            predict_pos = predicted_pos[0][0][0].detach().cpu().numpy()
            predict_next_pos = round(predict_pos[0])
            predict_next_pos = env_train.clip_position(predict_next_pos)
            next_cropped_plan = env_train.plan_withborder[:,
                            predict_next_pos - env_train.HALF_WINDOW_SIZE:predict_next_pos + env_train.HALF_WINDOW_SIZE + 1]
            local_memory.append((current_obs_normalized, action, reward, next_obs_normalized, current_cropped_plan, next_cropped_plan))
            reward_train += reward
            if total_steps % UPDATE_FREQ == 0:
                train_loss=agent.learning_process()
            if done:
                secs = int(time.time() - start_time)
                mins = secs / 60
                secs = secs % 60
                print(" | time in %d minutes, %d seconds\n" %(mins, secs))
                print('Epodise: ', episode,
                    '| Ep_reward: ', reward_train)
            if done:
                break
            prev_state = new_state
            current_pos = predict_next_pos
            hidden_batch, cell_batch = hidden_next, cell_next
            hidden_state, cell_state = hidden_state_next, cell_state_next
        agent.replay_memory.add_episode(local_memory)
        train_iou=env_train.iou()
        iou_test=0
        reward_test_total=0
        start_time_test = time.time()

        for _ in range(N_iteration_test):
            prev_state = env_val.reset()
            current_pos = prev_state[3]
            hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
            hidden_state, cell_state = agent.Eval_net.init_hidden_states(bsize=1)
            reward_test=0
            while True:
                current_cropped_plan = env_val.plan_withborder[:,
                            current_pos - env_val.HALF_WINDOW_SIZE:current_pos + env_val.HALF_WINDOW_SIZE + 1]
                current_pos = np.array(current_pos)
                current_obs = prev_state[0]
                current_obs_normalized = prev_state[1]
                action, hidden_state_next, cell_state_next = agent.choose_action(current_obs_normalized, current_cropped_plan, hidden_state, cell_state)
                new_state,reward,done = env_val.step(action)
                next_obs = new_state[1]
                torch_current_obs = torch.from_numpy(current_obs).view(1,1,7).float().to(device)
                torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
                torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
                torch_next_obs = torch.from_numpy(next_obs).view(1,1,7).float().to(device)
                input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)
                predicted_pos, hidden_next, cell_next = Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
                predict_pos = predicted_pos[0][0][0].detach().cpu().numpy()
                predict_next_pos = round(predict_pos[0])
                predict_next_pos = env_val.clip_position(predict_next_pos)
                next_cropped_plan = env_train.plan_withborder[:,
                            predict_next_pos - env_train.HALF_WINDOW_SIZE:predict_next_pos + env_train.HALF_WINDOW_SIZE + 1]
                reward_test += reward
                if done:
                    break
                prev_state = new_state
                current_pos = predict_next_pos
                hidden_batch, cell_batch = hidden_next, cell_next
                hidden_state, cell_state = hidden_state_next, cell_state_next
            reward_test_total+=reward_test
            iou_test+=env_val.iou()
        reward_test_total=reward_test_total/N_iteration_test
        IOU_test_total= iou_test/N_iteration_test
        secs = int(time.time() - start_time_test)
        mins = secs / 60
        secs = secs % 60
        print(" | time in %d minutes, %d seconds\n" %(mins, secs))
        print('Epodise: ', episode,
            '| Ep_reward_test:', reward_test_total)
        print('\nEpodise: ', episode,
            '| Ep_IOU_test: ', iou_test/N_iteration_test)
        if agent.greedy_epsilon > FINAL_EPSILON:
            agent.greedy_epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/N_iteration
            print("agent greedy_epsilon", agent.greedy_epsilon)
        if reward_test_total >= best_reward:
            torch.save(agent.Eval_net.state_dict(), model_dir+'Eval_net_episode_%d.pth' % (episode))
            torch.save(agent.Target_net.state_dict(), model_dir+'Target_net_episode_%d.pth' % (episode))
            best_reward=reward_test_total
        writer.add_scalars("log",
                                    {'train_loss': train_loss,
                                        'train_reward': reward_train,
                                        'train_iou': train_iou,
                                    'test_reward': reward_test_total,
                                    'test_iou':IOU_test_total,}, episode)
    writer.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    main(parser.parse_args())