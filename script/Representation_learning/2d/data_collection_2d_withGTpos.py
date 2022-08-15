import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__)) # 
sys.path.append(dir_path + '/../../../Env/2D/')
from DMP_Env_2D_static_normalized_GTpos import deep_mobile_printing_2d1r
import joblib
import numpy as np

replay_memory = []
env = deep_mobile_printing_2d1r(plan_choose=0)

for episode in range(30000):
    print(episode)
    obs = env.reset()
    local_memory = []
    while True:
        action = np.random.randint(5)
        new_obs, reward, done = env.step(action)
        local_memory.append((obs,action,new_obs))
        obs = new_obs
        if done:
            break
    replay_memory.append(local_memory)

save_path = "./"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
joblib.dump(replay_memory, save_path+"data_2d_static_dense_normalized_GTpos_30000.pkl")