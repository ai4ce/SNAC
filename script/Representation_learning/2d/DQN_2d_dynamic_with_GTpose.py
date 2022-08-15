import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_dynamic_GTpose import deep_mobile_printing_2d1r
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', help='device')
parser.add_argument('--model_dir', default="model_dir_DQN_2d_dynamic_dense_with_GTpose", type=str, help='The path to the saved model')
parser.add_argument('--log_dir', default="log_dir_DQN_2d_dynamic_dense_with_GTpose", type=str, help='The path to log')
parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size')
parser.add_argument('--plan_type', default=0, type=int, help='dense:0; sparse:1')
parser.add_argument('--Random_seed', default=1, type=int, help='random seed')
parser.add_argument('--Replay_buffer_size', default=500, type=int, help='replay buffer size')
parser.add_argument('--N_iteration', default=3000, type=int, help='Number of tarining iteration') 
args = parser.parse_args()

## hyper parameter
seeds=args.Random_seed
set_seed(seeds)
minibatch_size=args.batch_size
Lr=args.lr
N_iteration=args.N_iteration
N_iteration_test=10
alpha=0.9
Replay_memory_size=args.Replay_buffer_size
Update_traget_period=200
UPDATE_FREQ=1
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0
PALN_CHOICE=args.plan_type  ##0: dense 1: sparse
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
PLAN_LIST=["dense","sparse"]
PLAN_NAME=PLAN_LIST[PALN_CHOICE]
OUT_FILE_NAME="DQN_2d_dynamic_withGTpose_"+PLAN_NAME+"_lr"+str(Lr)+"_seed_"+str(seeds)
print(OUT_FILE_NAME)
model_dir=args.model_dir+"/"+OUT_FILE_NAME+"/"
env = deep_mobile_printing_2d1r(plan_choose=PALN_CHOICE)
if os.path.exists(model_dir)==False:
    os.makedirs(model_dir)

Action_dim=env.action_dim
State_dim=env.state_dim
plan_dim=env.plan_width
print("state_dim",State_dim)
print("total_step",env.total_step)
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
        self.fc_1 = get_and_init_FC_layer(State_dim+3+32, 64)
        self.fc_2 = get_and_init_FC_layer(64, 128)
        self.fc_3 = get_and_init_FC_layer(128, 128)
        self.out = get_and_init_FC_layer(128, 1)
        self.relu = nn.ReLU()
    def forward(self, s, pose, a,plan,bsize):
        plan = plan.view(bsize,1,20,20) 
        conv_out = self.conv_layer1(plan)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_layer2(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_layer3(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = conv_out.view(bsize,32)
        x=torch.cat((s, pose, a,conv_out),dim=1)
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
    def store_memory(self,s,a,r, s_next, pose_current, pose_next, env_plan):
        self.replay_memory.append((s,a,r,s_next,pose_current, pose_next, env_plan))
        self.count_memory+=1
    def choose_action(self, s, pose, env_plan):      
        env_plan=torch.from_numpy(env_plan).float().to(device)
        env_plan=env_plan.unsqueeze(0)
        state = torch.FloatTensor(s).to(device)
        pose = torch.FloatTensor(pose).unsqueeze(0).to(device)
        choose=np.random.uniform()
        Q=[]
        if choose<=self.greedy_epsilon:
            action=np.random.randint(0, Action_dim)         
        else:
            for item in range(Action_dim):
                a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
                Q_value=self.Eval_net(state, pose, a, env_plan, 1)
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
        current_pose = []
        acts = []
        rewards = []
        next_states = []
        next_pose = []
        env_plans = []

        for b in minibatch:
            current_states.append(b[0])
            current_pose.append(b[4])
            acts.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])   
            next_pose.append(b[5])
            env_plans.append(b[6])
        current_states = np.array(current_states)
        current_pose = np.array(current_pose)
        acts = np.array(acts).reshape(minibatch_size,1)
        rewards = np.array(rewards).reshape(minibatch_size,1)
        next_states = np.array(next_states)
        next_pose = np.array(next_pose)
        env_plans = np.array(env_plans)
        b_state = torch.from_numpy(current_states).float().to(device).squeeze(1)
        b_current_pose = torch.from_numpy(current_pose).float().to(device).squeeze(1)
        b_action = torch.from_numpy(acts).float().to(device)
        b_reward = torch.from_numpy(rewards).float().to(device)
        b_state_next = torch.from_numpy(next_states).float().to(device).squeeze(1)
        b_next_pose = torch.from_numpy(next_pose).float().to(device).squeeze(1)
        b_env_plans = torch.from_numpy(env_plans).float().to(device)
        
        Q_eval = self.Eval_net(b_state,b_current_pose, b_action, b_env_plans, minibatch_size)
        Q=torch.zeros(minibatch_size,Action_dim).to(device)
        for item in range(Action_dim):
            a=torch.FloatTensor(np.array([item])).unsqueeze(0).to(device)
            a=a.expand(minibatch_size,-1)
            Q_value=self.Target_net(b_state_next,b_next_pose,a, b_env_plans, minibatch_size).detach().squeeze(1)
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
    obs = env.reset()
    env_plan=obs[1]
    
    while True:
        current_obs = obs[0]
        current_pos = obs[2]
        action = np.random.randint(0,Action_dim)
        new_obs, reward, done = env.step(action)
        next_obs = new_obs[0]
        next_pos = new_obs[2]
        agent.store_memory(current_obs, action, reward, next_obs, current_pos, next_pos, env_plan)
        obs = new_obs
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
    obs = env.reset()
    env_plan=obs[1]
    print("total_brick",env.total_brick)
    reward_train = 0
    start_time = time.time()
    while True:
        total_steps +=1
        current_obs = obs[0]
        current_pos = obs[2]
        
        action = agent.choose_action(current_obs, current_pos,env_plan)
        new_obs, reward, done = env.step(action)

        next_obs = new_obs[0]
        next_pos = new_obs[2]
        
        agent.store_memory(current_obs, action, reward, next_obs, current_pos, next_pos, env_plan)

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
        obs = new_obs
    train_iou=iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)

    iou_test=0
    reward_test_total=0
    start_time_test = time.time()

    for _ in range(N_iteration_test):
        obs = env.reset()
        env_plan=obs[1]
        reward_test=0
        while True:
            current_obs = obs[0]
            current_pos = obs[2]
            
            action = agent.choose_action(current_obs, current_pos, env_plan)

            new_obs, reward, done = env.step(action)            

            reward_test += reward
            if done:
                break
            obs = new_obs
        reward_test_total+=reward_test
        iou_test+=iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
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
        with open(model_dir+"brick.pickle", "wb") as fp: 
            pickle.dump(env.brick_memory, fp)
        with open(model_dir+"position.pickle", "wb") as fp: 
            pickle.dump(env.position_memory, fp)
        best_reward=reward_test_total
    writer.add_scalars("log",
                                   {'train_loss': train_loss,
                                    'train_reward': reward_train,
                                    'train_iou': train_iou,
                                   'test_reward': reward_test_total,
                                   'test_iou':IOU_test_total,}, episode)
JSON_log_PATH="./JSON/"
if os.path.exists(JSON_log_PATH)==False:
    os.makedirs(JSON_log_PATH)                                   
writer.export_scalars_to_json(JSON_log_PATH+OUT_FILE_NAME+".json")
writer.close()

    

    









