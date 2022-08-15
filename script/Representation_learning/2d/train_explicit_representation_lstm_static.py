import model
import joblib
from collections import deque
import torch
import torch.nn as nn 
import random
import numpy as np
import argparse
import os 
from scipy.stats import multivariate_normal
from torch.utils.tensorboard import SummaryWriter
import time

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

def trainer(args, batch, model, optimizer, device):
    if args.loss_type == "L2":
        criteria = nn.MSELoss()
    else:
        criteria = nn.KLDivLoss(reduction="batchmean")
    hidden_batch, cell_batch = model.init_hidden_states(bsize=args.batch_size)
    current_obs = []
    next_obs = []
    acts = []
    current_pos = []
    next_pos = []
    for b in batch:
        cobs, nobs, ac, cpos, npos = [], [], [], [], []
        for element in b:
            cobs.append(element[0][0])
            nobs.append(element[2][0])
            ac.append(element[1])
            cpos.append(element[0][1])
            npos.append(element[2][1])
        current_obs.append(cobs)
        next_obs.append(nobs)
        acts.append(ac)
        current_pos.append(cpos)
        next_pos.append(npos)

    current_obs = np.array(current_obs)
    next_obs = np.array(next_obs)
    acts = np.array(acts)
    current_pos = np.array(current_pos)
    next_pos = np.array(next_pos)
   
    torch_current_obs = torch.from_numpy(current_obs).float().squeeze(2).to(device)
    torch_next_obs = torch.from_numpy(next_obs).float().squeeze(2).to(device)
    torch_actions = torch.from_numpy(acts).float().unsqueeze(-1).to(device)
    torch_current_pos = torch.from_numpy(current_pos).float().to(device)
    if args.loss_type == "L2":
        torch_next_pos = torch.from_numpy(next_pos).float().to(device)
    else:
        x, y = np.mgrid[0:26,0:26]
        area_filed = np.dstack((x, y))
        next_pos_list = []
        for i in range(args.batch_size):
            next_pos_list_one_batch = []
            for j in range(args.Time_step):
                rv = multivariate_normal([next_pos[i,j,0], next_pos[i,j,1]], [[2, 0], [0, 2]])
                heatmap=rv.pdf(area_filed)
                next_pos_list_one_batch.append(heatmap)
            next_pos_list.append(next_pos_list_one_batch)
        next_pos_list = np.array(next_pos_list)
        torch_next_pos = torch.from_numpy(next_pos_list).float().to(device)
    input_torch_data = torch.cat((torch_current_obs,torch_next_obs,torch_actions),dim=2)
    predicted_pos, hidden_state, cell_state = model(input_torch_data, torch_current_pos, hidden_batch, cell_batch)
    loss = 0 
    for i in range(args.Time_step):
        loss += criteria(predicted_pos[i],torch_next_pos[:,i:i+1,:])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
   

def main(args):
    replaymemory=Memory(args.Replay_buffer_size)
<<<<<<< HEAD:script/Representation_learning/2d/train_explicit_representation_lstm.py
    filename = './data_2d_static_dense_normalized_GTpos_30000.pkl'
=======
    filename = '/mnt/NAS/home/WenyuHan/SNAC/Representation/2D/data_2d_static_sparse_normalized_GTpos_30000.pkl'
>>>>>>> 7c6ca21ee5bd4ee1f9f19d1bdc7c289bedd02710:script/Representation_learning/2d/train_explicit_representation_lstm_static.py
    local_memories = joblib.load(filename)
    for local_memory in local_memories:
        replaymemory.add_episode(local_memory)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUT_FILE_NAME = "/Lr_"+str(args.lr)+"_batchsize_"+str(args.batch_size)+"/"
    log_dir=args.log_dir +  OUT_FILE_NAME
    model_dir = args.model_dir + OUT_FILE_NAME
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    Lnet = model.SNAC_Lnet(input_size=105, hidden_size=args.hidden_size, device=device, Loss_type=args.loss_type)
    Lnet.to(device)
    Lnet.train()
    optimizer = torch.optim.Adam(Lnet.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir)
    for episode in range(args.N_iteration):
        start_time = time.time()
        batch = replaymemory.get_batch(bsize=args.batch_size,time_step=args.Time_step)  
        loss_total = trainer(args, batch, Lnet, optimizer, device)
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        print('Epodise: ', episode+1,
            '| total_loss:', loss_total)
        print(" | time in %d minutes, %d seconds\n" % (mins, secs))
        writer.add_scalars("log",
                                {'loss_total': loss_total,}, episode)
        if (episode + 1) % args.checkpoint_freq == 0 or episode + 1 == args.N_iteration:
            # Save model
            model_name = 'model_{:08d}.pth'.format(episode + 1)
            torch.save(Lnet.state_dict(), model_dir + model_name)
            
    writer.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
<<<<<<< HEAD:script/Representation_learning/2d/train_explicit_representation_lstm.py
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--model_dir', default="model_dir_explict_representation_lstm", type=str, help='The path to the saved model')
    parser.add_argument('--log_dir', default="log_dir_explict_representation_lstm", type=str, help='The path to log')
=======
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--model_dir', default="model_dir_explict_representation_lstm_DQN2d_static_sparse", type=str, help='The path to the saved model')
    parser.add_argument('--log_dir', default="log_dir_explict_representation_lstm_DQN2d_static_sparse", type=str, help='The path to log')
>>>>>>> 7c6ca21ee5bd4ee1f9f19d1bdc7c289bedd02710:script/Representation_learning/2d/train_explicit_representation_lstm_static.py
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=500, type=int, help='Batch size')
    parser.add_argument('--Time_step', default=20, type=int, help='sequence length')
    parser.add_argument('--hidden_size', default=128, type=int, help='LSTM hidden state size')
    parser.add_argument('--Replay_buffer_size', default=30000, type=int, help='replay buffer size')
    parser.add_argument('--N_iteration', default=1000000, type=int, help='Number of tarining iteration') 
    parser.add_argument('--checkpoint_freq', default=5000, type=int, help='checkpoint saved frequency')    
    parser.add_argument('--loss_type', default="L2", type=str, help='choose loss type from L2 and KL')    
    
    args = parser.parse_args()
    print(args)
    main(args)