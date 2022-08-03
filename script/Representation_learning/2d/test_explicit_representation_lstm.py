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
import sys
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_static_normalized_GTpos import deep_mobile_printing_2d1r

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Lnet = model.SNAC_Lnet(input_size=105, hidden_size=128, device=device, Loss_type="L2")
model_checkpoint = torch.load("./model_01000000.pth", map_location=device)

Lnet.load_state_dict(model_checkpoint)
env = deep_mobile_printing_2d1r(plan_choose=0)
Lnet.to(device)
Lnet.eval()
save_test_results_image_path = Path("./test_images_explicit")
if not save_test_results_image_path.exists():
    save_test_results_image_path.mkdir()

for episode in range(1):
    step = 1
    hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
    obs = env.reset()
    current_pos = obs[1]
    while True:
        pos_map = np.zeros((26,26,3))
        # predict_pos_map = np.zeros((26,26))
        current_obs = obs[0]
        action = np.random.randint(5)
        current_pos = np.array(current_pos)
        torch_current_obs = torch.from_numpy(current_obs).float().unsqueeze(0).to(device)
        torch_current_pos = torch.from_numpy(current_pos).float().unsqueeze(0).to(device)
        torch_current_pos = torch_current_pos.unsqueeze(0)
        
        new_obs, reward, done = env.step(action)
        torch_action = torch.from_numpy(np.array(action).reshape(1,1,1)).float().to(device)

        next_pos = new_obs[1]
        next_obs = new_obs[0]

        torch_next_obs = torch.from_numpy(next_obs).float().unsqueeze(0).to(device)
        
        # print("torch_current_obs",torch_current_obs.shape)
        # print("torch_next_obs",torch_next_obs.shape)
        # print("torch_current_pos",torch_current_pos.shape)
        # print("action",torch_action.shape)

        input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action),dim=2)
        # print(input_data.shape)

        predicted_pos, hidden_next, cell_next =Lnet(input_data,torch_current_pos, hidden_batch, cell_batch)
        predict_pos = predicted_pos[0].squeeze().detach().cpu().numpy()
        predict_next_pos = [0,0]
        predict_next_pos[0] = round(predict_pos[0])  
        predict_next_pos[1] = round(predict_pos[1])  

        print("GT pos",next_pos)
        print("predict pos",predict_next_pos)
        print("action",action)

        pos_map[next_pos[0],next_pos[1],0] = 1
        pos_map[predict_next_pos[0],predict_next_pos[1],1] = 1

        if step % 10 == 0:
            # f, axarr = plt.subplots(1,2) 
            plt.imshow(pos_map)
            # axarr[0].axis('off')
            # axarr[1].imshow(pos_map)
            # axarr[1].axis('off')
            plt.savefig(str(save_test_results_image_path)+"/timestep_"+str(step)+'.jpeg')
            plt.close()
        obs = new_obs
        current_pos = predict_next_pos
        hidden_batch, cell_batch = hidden_next, cell_next
        step+=1
        if done:
            break