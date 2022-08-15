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
from DMP_Env_2D_static_normalized import deep_mobile_printing_2d1r
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AEmodel = model.RecurrentAE(input_size=51, output_size=49*3+2, hidden_size=512, device=device)
model_checkpoint_encoder = torch.load("./model_encoder01000000.pth", map_location=device)
model_checkpoint_decoder = torch.load("./model_decoder01000000.pth", map_location=device)

AEmodel.encoder.load_state_dict(model_checkpoint_encoder)
AEmodel.decoder.load_state_dict(model_checkpoint_decoder)
env = deep_mobile_printing_2d1r(plan_choose=0)
AEmodel.eval()
save_test_results_image_path = Path("./test_images")
if not save_test_results_image_path.exists():
    save_test_results_image_path.mkdir()
for episode in range(1):
    step = 1
    hidden_batch = AEmodel.encoder.init_hidden_states(bsize=1)
    obs = env.reset()
    while True:
        f, axarr = plt.subplots(1,2) 
        GT_obs = obs[:,0:49].reshape(7,7)
        action = np.random.randint(5)
        torch_current_obs = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        predicted_obs, hidden_next =AEmodel(torch_current_obs,hidden_batch)

        predicted = predicted_obs[:,:,:147].view(49,3).detach().cpu().numpy() 
        
        predicted = np.argmax(predicted,axis=1).reshape(7,7)

        axarr[0].imshow(GT_obs)
        axarr[0].axis('off')
        axarr[1].imshow(predicted)
        axarr[1].axis('off')
        plt.savefig(str(save_test_results_image_path)+"/timestep_"+str(step)+'.jpeg')
        plt.close()
    
        new_obs, reward, done = env.step(action)
        obs = new_obs
        step+=1
        hidden_batch = hidden_next
        if done:
            break