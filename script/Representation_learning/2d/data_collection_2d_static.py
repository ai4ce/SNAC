import sys
sys.path.append('../../../Env/2D/')
from DMP_Env_2D_static_normalized import deep_mobile_printing_2d1r
import joblib
import os
import numpy as np

replay_memory = []
env = deep_mobile_printing_2d1r(plan_choose=0)

for episode in range(100):
    print(episode)
    obs = env.reset()
    local_memory = []
    while True:
        action = np.random.randint(5)
        new_obs, reward, done = env.step(action)
        local_memory.append((obs))
        obs = new_obs
        if done:
            break
    replay_memory.append(local_memory)

# save_path = "/mnt/NAS/home/WenyuHan/SNAC/Representation/2D/"
save_path = "./"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
joblib.dump(replay_memory, save_path+"data_2d_static_sparse_normalized_30000.pkl")
