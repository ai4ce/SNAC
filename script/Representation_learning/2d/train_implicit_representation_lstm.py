import model
import joblib
from collections import deque
import torch
import torch.nn as nn 
import random
import numpy as np
import argparse
import os 
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
    loss_crossentropy = nn.CrossEntropyLoss()
    loss_L2 = nn.MSELoss()
    hidden_batch = model.encoder.init_hidden_states(bsize=args.batch_size)
    current_states = []
    acts = []
    for b in batch:
        cs= []
        for element in b:
            cs.append(element[0])
        current_states.append(cs)
    current_states = np.array(current_states)
   
    torch_current_states = torch.from_numpy(current_states).float().squeeze(2).to(device)
   
    predicted = model(torch_current_states,hidden_batch)
    predicted_obs = predicted[:,:,:147].view(args.batch_size,args.Time_step,49,3)
    predicted_obs = predicted_obs.permute(0, 3, 1, 2) ## make the order (N,C,L,d2)
   
    target_obs = torch_current_states[:,:,:49].long() ## make the order (N,L,d2)
    

    predicted_value = predicted[:,:,147:149]
    
    target_value = torch_current_states[:,:,49:51]
   
    loss_1 = loss_crossentropy(predicted_obs,target_obs) 
    loss_2 = loss_L2(predicted_value,target_value)
    loss = loss_1 + loss_2 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), loss_1.item(), loss_2.item()
   

def main(args):
    replaymemory=Memory(args.Replay_buffer_size)
    filename = './data_2d_static_sparse_normalized_30000.pkl'
    local_memories = joblib.load(filename)
    for local_memory in local_memories:
        replaymemory.add_episode(local_memory)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUT_FILE_NAME = "/Lr_"+str(args.lr)+"_batchsize_"+str(args.batch_size)+"/"
    log_dir=args.log_dir +  OUT_FILE_NAME
    model_dir = args.model_dir + OUT_FILE_NAME
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    AEmodel = model.RecurrentAE(input_size=51, output_size=49*3+2, hidden_size=args.hidden_size, device=device)
    optimizer = torch.optim.Adam(AEmodel.parameters(), lr=args.lr)
    batch = replaymemory.get_batch(bsize=args.batch_size,time_step=args.Time_step)  
    writer = SummaryWriter(log_dir)
    for episode in range(args.N_iteration):
        start_time = time.time()
        loss_total, loss_obs_image, loss_obs_value = trainer(args, batch, AEmodel, optimizer, device)
        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        print('Epodise: ', episode+1,
            '| total_loss:', loss_total, '| loss_obs_image:',loss_obs_image, '| loss_obs_value:',loss_obs_value)
        print(" | time in %d minutes, %d seconds\n" % (mins, secs))
        writer.add_scalars("log",
                                {'loss_total': loss_total,
                                'loss_obs_image': loss_obs_image,
                                'loss_obs_value': loss_obs_value}, episode)
        if (episode + 1) % args.checkpoint_freq == 0 or episode + 1 == args.N_iteration:
            # Save model
            model_name_encoder = 'model_encoder{:08d}.pth'.format(episode + 1)
            model_name_decoder = 'model_decoder{:08d}.pth'.format(episode + 1)
            torch.save(AEmodel.encoder.state_dict(), model_dir + model_name_encoder)
            torch.save(AEmodel.decoder.state_dict(), model_dir + model_name_decoder)
    writer.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--model_dir', default="model_dir_implict_representation_lstm", type=str, help='The path to the saved model')
    parser.add_argument('--log_dir', default="log_dir_implict_representation_lstm", type=str, help='The path to log')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('--Time_step', default=50, type=int, help='sequence length')
    parser.add_argument('--hidden_size', default=256, type=int, help='LSTM hidden state size')
    parser.add_argument('--Replay_buffer_size', default=30000, type=int, help='replay buffer size')
    parser.add_argument('--N_iteration', default=10000, type=int, help='Number of tarining iteration') 
    parser.add_argument('--checkpoint_freq', default=1000, type=int, help='checkpoint saved frequency')    
    args = parser.parse_args()
    print(args)
    main(args)
