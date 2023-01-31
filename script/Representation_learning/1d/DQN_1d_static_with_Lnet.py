import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append('../../../Env/1D/')
from DMP_Env_1D_static_Lnet import deep_mobile_printing_1d1r
sys.path.append('../../../')
import utils
import pickle
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
    # hyper parameter
    seeds=args.Random_seed
    set_seed(seeds)
    minibatch_size=args.batch_size
    Lr=args.lr
    N_iteration=args.N_iteration
    N_iteration_test=10
    alpha=0.9
    Replay_memory_size=args.Replay_buffer_size
    Update_traget_period=args.update_traget_period
    UPDATE_FREQ = 1
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    PALN_CHOICE=args.plan_type  ##0: sin 1: Guassian 2:step
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    PLAN_LIST=["sin","gaussian","step"]
    PLAN_NAME=PLAN_LIST[PALN_CHOICE]
    OUT_FILE_NAME="DQN_1d_static_Lnet_"+PLAN_NAME+"_lr"+str(Lr)+"_seed_"+str(seeds)
    print(OUT_FILE_NAME)
    model_dir=args.model_dir+"/"+OUT_FILE_NAME+"/"
    if os.path.exists(model_dir)==False:
        os.makedirs(model_dir)
    env = deep_mobile_printing_1d1r(plan_choose=PALN_CHOICE)
    Action_dim=env.action_dim
    State_dim=env.state_dim
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
            #self.fc_1 = get_and_init_FC_layer(State_dim+1, 64) # should this just be 6? obs = 5, pos = 1
            self.fc_1 = get_and_init_FC_layer(State_dim+1+1, 64) # should this just be 7? obs = 5, pos = 1, act = 1
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
    Lnet = model.SNAC_Lnet(input_size=16, hidden_size=128, device=device, Loss_type="L2")
    model_checkpoint = torch.load(args.pretrainmodel, map_location=device)
    Lnet.load_state_dict(model_checkpoint)
    Lnet.to(device)
    Lnet.eval()
    agent=DQN_AGNET(device)
    number_steps=0
    while True:
        hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
        prev_state = env.reset()
        obs = np.reshape(prev_state, (1,1,8))
        current_pos = obs[0][0][7:] #[0]
        while True:           
            # action = np.random.randint(0, 3, (1,))
            current_pos = np.array(current_pos)
            current_obs = obs[0][0][0:7]
            action = np.random.randint(0,Action_dim)
            new_obs,reward,done = env.step(action)
            torch_current_obs = torch.from_numpy(current_obs).view(1,1,7).float().to(device)
            torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
            torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
            new_obs = np.reshape(new_obs, (1,1,8))
            next_obs = new_obs[0][0][0:7]
            torch_next_obs = torch.from_numpy(next_obs).view(1,1,7).float().to(device)
            input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)
            predicted_pos, hidden_next, cell_next = Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
            predict_pos = predicted_pos[0][0][0].detach().cpu().numpy()
            predict_next_pos = round(predict_pos[0])
            agent.store_memory(current_obs, action, reward, next_obs, current_pos, predict_next_pos)
            obs = new_obs
            current_pos = predict_next_pos
            hidden_batch, cell_batch = hidden_next, cell_next
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
        hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
        reward_train = 0
        prev_state = env.reset()
        obs = np.reshape(prev_state, (1,1,8))
        current_pos = obs[0][0][7:] #[0]
        start_time = time.time()
        while True:
            total_steps +=1
            current_obs = obs[0][0][0:7]
            # current_pos = np.array([float(current_pos)])
            action = agent.choose_action(current_obs, current_pos)
            new_obs,reward,done = env.step(action)
            torch_current_obs = torch.from_numpy(current_obs).view(1,1,7).float().to(device)
            torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
            torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
            new_obs = np.reshape(new_obs, (1,1,8))
            next_obs = new_obs[0][0][0:7]
            torch_next_obs = torch.from_numpy(next_obs).view(1,1,7).float().to(device)
            input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)
            predicted_pos, hidden_next, cell_next = Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
            predict_pos = predicted_pos[0][0][0].detach().cpu().numpy()
            predict_next_pos = round(predict_pos[0])
            agent.store_memory(current_obs, action, reward, next_obs, current_pos, predict_next_pos)
            reward_train += reward
            if total_steps % UPDATE_FREQ == 0:
                train_loss=agent.learning_process()   
            if done:
                secs = int(time.time() - start_time)
                mins = secs / 60
                secs = secs % 60
                print(" | time in %d minutes, %d seconds\n" %(mins, secs))
                print('Epodise: ', episode,
                    '| Ep_reward_train: ', reward_train)
            if done:
                break
            obs = new_obs
            current_pos = np.array([float(predict_next_pos)])
            hidden_batch, cell_batch = hidden_next, cell_next
        train_iou=env.iou()
        iou_test=0
        reward_test_total=0
        start_time_test = time.time()
        for _ in range(N_iteration_test):
            reward_test=0
            hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
            prev_state = env.reset()
            obs = np.reshape(prev_state, (1,1,8))
            current_pos = obs[0][0][7:] #[0]
            while True:
                current_obs = obs[0][0][0:7]
                # current_pos = np.array([float(current_pos)])
                action = agent.choose_action(current_obs, current_pos)
                new_obs,reward,done = env.step(action)
                torch_current_obs = torch.from_numpy(current_obs).view(1,1,7).float().to(device)
                torch_current_pos = torch.from_numpy(current_pos).view(1,1,1).float().to(device)
                torch_action = torch.from_numpy(np.array(action)).view(1,1,1).float().to(device)
                new_obs = np.reshape(new_obs, (1,1,8))
                next_obs = new_obs[0][0][0:7]
                torch_next_obs = torch.from_numpy(next_obs).view(1,1,7).float().to(device)
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
            '| Ep_IOU_test: ', iou_test/N_iteration_test)
        if agent.greedy_epsilon > FINAL_EPSILON:
            agent.greedy_epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/N_iteration
            print("agent greedy_epsilon", agent.greedy_epsilon)
        if reward_test_total >= best_reward:
            torch.save(agent.Eval_net.state_dict(), model_dir+'Eval_net_episode_%d.pth' % (episode))
            torch.save(agent.Target_net.state_dict(), model_dir+'Target_net_episode_%d.pth' % (episode))
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