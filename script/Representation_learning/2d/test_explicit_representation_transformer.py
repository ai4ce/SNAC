import model
import joblib
from collections import deque
import torch
import torch.nn as nn 
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
import matplotlib.pyplot as plt
from pathlib import Path
import os 
import sys
dir_path = os.path.dirname(os.path.realpath(__file__)) # 
sys.path.append(dir_path + '/../../../Env/2D/')
from DMP_Env_2D_static_normalized_GTpos import deep_mobile_printing_2d1r

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# AEmodel = model.AttentionAE(
#         in_features=105,
#         out_features=2,
#         nhead=15,
#         num_layers=10).to(device)
AEmodel = model.AttentionAE(
        in_features=105,
        out_features=2,
        #max_len=1,
        max_len=20,
        nhead=15,
        num_layers=10).to(device)
#model_checkpoint = torch.load("./model_dir_implict_representation_vit/Lr_0.0001_batchsize_10_dense/dense_model_01000000.pth", map_location=device)
model_checkpoint = torch.load("./model_dir_explicit_representation_transformer_pos/Lr_0.0001_batchsize_50_dense/dense_model_01000000.pth", map_location=device)

AEmodel.load_state_dict(model_checkpoint)
env = deep_mobile_printing_2d1r(plan_choose=0)
AEmodel.eval()
save_test_results_image_path = Path("./test_images_explicit_transformer_dense")
#save_test_results_image_path = Path("./test_images_explicit_sparse")
if not save_test_results_image_path.exists():
    save_test_results_image_path.mkdir()

for episode in range(1):
    step = 1
    # hidden_batch, cell_batch = Lnet.init_hidden_states(bsize=1)
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
        input_data = torch.cat((torch_current_obs,torch_next_obs,torch_action,torch_current_pos),dim=2)

        _, predicted_pos = AEmodel(input_data) # 1x1x2
        predict_pos = predicted_pos[0].squeeze().detach().cpu().numpy()
        print(predict_pos)
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
        step+=1
        if done:
            break