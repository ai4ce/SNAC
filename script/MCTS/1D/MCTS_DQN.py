#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 18:19:21 2021

@author: hanwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from collections import deque
import random
import sys
sys.path.append('../../../Env/1D/')
import gym
from DMP_Env_1D_static_MCTS import deep_mobile_printing_1d1r_MCTS
import os
import uct
import numpy as np
import time 


log_path="./final_test/"

env = deep_mobile_printing_1d1r_MCTS(plan_choose=2)

if os.path.exists(log_path)==False:
    os.makedirs(log_path)

## hyper parameter
minibatch_size=2000
Lr=0.001
N_iteration=3000
N_iteration_test=10
alpha=0.9
Replay_memory_size=50000
Update_traget_period=200
Action_dim=env.action_dim
State_dim=env.state_dim
UPDATE_FREQ=1
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0

print("state_dim",State_dim)
print("total_step",env.total_step)

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
        # self.replay_memory = np.zeros((Replay_memory_size, State_dim * 2 + 2)) 
        self.replay_memory = deque(maxlen=Replay_memory_size)    
        self.optimizer = torch.optim.Adam(self.Eval_net.parameters(), lr=Lr)
        self.greedy_epsilon=0.2
        self.loss = nn.SmoothL1Loss()
        self.loss_his=[]
    def store_memory(self,s,a,r,s_next):
        # transition=np.concatenate((s,[a],[r],s_next))
        # index=self.count_memory%Replay_memory_size
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
        
        acts = np.array(acts).reshape(minibatch_size,1)
        
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
        self.loss_his.append(loss.item())