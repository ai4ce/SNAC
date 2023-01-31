import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../../../Env/3D/')
sys.path.append('../../../')
import utils
from DMP_simulator_3d_static_circle import deep_mobile_printing_3d1r
import pickle
import time
import os
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter
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
    UPDATE_FREQ=1
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    PALN_CHOICE=args.plan_type   ##0: dense 1: sparse
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    PLAN_LIST=["dense","sparse"]
    PLAN_NAME=PLAN_LIST[PALN_CHOICE]
    OUT_FILE_NAME="DQN_3d_"+PLAN_NAME+"_lr"+str(Lr)+"_seed_"+str(seeds)
    print(OUT_FILE_NAME)
    model_dir=args.model_dir+"/"+OUT_FILE_NAME+"/"
    env = deep_mobile_printing_3d1r(plan_choose=PALN_CHOICE)
    if os.path.exists(model_dir)==False:
        os.makedirs(model_dir)
    Action_dim=env.action_dim
    State_dim=env.state_dim
    plan_dim=env.plan_width
    print("state_dim",State_dim)
    print("total_step",env.total_step)
    print("Action_dim",Action_dim)

    def get_and_init_FC_layer(din, dout):
        li = nn.Linear(din, dout)
        nn.init.xavier_uniform_(
        li.weight.data, gain=nn.init.calculate_gain('relu'))
        li.bias.data.fill_(0.)
        return li

    class Q_NET(nn.Module):
        def __init__(self, ):
            super(Q_NET, self).__init__()
            self.fc_1 = get_and_init_FC_layer(State_dim+1, 64)
            self.fc_2 = get_and_init_FC_layer(64, 128)
            self.fc_3 = get_and_init_FC_layer(128, 128)
            self.out = get_and_init_FC_layer(128, 1)
            self.relu = nn.ReLU()
        def forward(self, s,a):
            x=torch.cat((s,a),dim=1)
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
            self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
            self.greedy_epsilon=0.2
            self.loss = nn.SmoothL1Loss()
            self.loss_his=[]
        def store_memory(self,s,a,r,s_next):
            self.replay_memory.append((s,a,r,s_next))
            self.count_memory+=1
        def choose_action(self,s):
            state=torch.FloatTensor(s).to(device)       
            choose=np.random.uniform()
            Q=[]
            if choose<=self.greedy_epsilon:
                action=np.random.randint(0, Action_dim)         
            else:
                for item in range(Action_dim):
                    a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                    Q_value=self.Eval_net(state,a)
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
            for b in minibatch:
                current_states.append(b[0])
                acts.append(b[1])
                rewards.append(b[2])
                next_states.append(b[3])   
            current_states = np.array(current_states)
            # print("current_states_shape",current_states.shape)
            acts = np.array(acts).reshape(minibatch_size,1)
            # print("action_shape",acts.shape)
            rewards = np.array(rewards).reshape(minibatch_size,1)
            next_states = np.array(next_states)
            b_state = torch.from_numpy(current_states).float().to(device).squeeze(1)
            b_action = torch.from_numpy(acts).float().to(device)
            b_reward = torch.from_numpy(rewards).float().to(device)
            b_state_next = torch.from_numpy(next_states).float().to(device).squeeze(1)
            Q_eval = self.Eval_net(b_state,b_action)
            Q=torch.zeros(minibatch_size,Action_dim).to(device)
            for item in range(Action_dim):
                a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                a=a.expand(minibatch_size,-1)
                Q_value=self.Target_net(b_state_next,a).detach().squeeze(1)
                Q[:,item]=Q_value 
            Q_next=torch.max(Q,dim=1)[0].reshape(minibatch_size,1)
            Q_target = b_reward + alpha * Q_next
            loss = self.loss(Q_eval, Q_target)
            loss.backward()
            self.optimizer.step()
            self.learn_step+=1
            train_loss=loss.item()
            return train_loss

    agent=DQN_AGNET(device)
    number_steps=0
    while True:
        prev_state = env.reset()
        while True:
            action = np.random.randint(0,Action_dim)
            next_state,reward,done = env.step(action)
            agent.store_memory(prev_state, action, reward, next_state)
            prev_state = next_state
            number_steps+=1
            if done:
                break
            if number_steps==Replay_memory_size:
                break
        if number_steps==Replay_memory_size:
                break

    agent.greedy_epsilon=INITIAL_EPSILON
    best_reward=0
    total_steps = 0
    writer = SummaryWriter(args.log_dir+"/"+OUT_FILE_NAME)
    for episode in range(N_iteration):
        state = env.reset()
        print("total_brick",env.total_brick)
        reward_train = 0
        start_time = time.time()
        while True:
            total_steps +=1
            action = agent.choose_action(state)
            state_next, r, done = env.step(action)
            agent.store_memory(state, action, r, state_next)
            reward_train += r
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
            state = state_next
        train_iou=env.iou()
        iou_test=0
        reward_test_total=0
        start_time_test = time.time()
        for _ in range(N_iteration_test):
            state = env.reset()
            reward_test=0
            while True:
                action = agent.choose_action(state)
                state_next, r, done = env.step(action)
                reward_test += r
                if done:
                    break
                state = state_next
            reward_test_total+=reward_test
            iou_test+=env.iou()
        reward_test_total=reward_test_total/N_iteration_test
        IOU_test_total= iou_test/N_iteration_test
        secs = int(time.time() - start_time_test)
        mins = secs / 60
        secs = secs % 60
        print(" | time in %d minutes, %d seconds\n" %(mins, secs))
        print('Epodise: ', episode,
            '| Ep_reward_test:', reward_test_total)
        print('\nEpodise: ', episode,
            '| Ep_IOU_test: ', IOU_test_total)
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

    

    










        

        









