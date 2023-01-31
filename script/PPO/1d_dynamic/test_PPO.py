from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import time
import DMP_Env_1D_dynamic_usedata_plan
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf
import os
print("successful load module")

log_path = "./results.pth"
test_agent = PPO2.load(log_path)
iou_all_average = 0
iou_all_min = 1
env = DMP_Env_1D_dynamic_usedata_plan.deep_mobile_printing_1d1r(data_path="./data_1d_dynamic_sin_envplan_500_test.pkl")
N_iteration_test = 500
best_iou = 0
iou_test_total = 0
iou_min = 1
reward_test_total = 0
start_time_test = time.time()
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
for ep in range(N_iteration_test):
    obs = env.reset()
    reward_test = 0
    while True:
        action, _ = test_agent.predict(obs)
        obs, r, done, info = env.step(action)
        reward_test += r
        if done:
            break
    iou_test = env.iou()
    iou_min = min(iou_min, iou_test)
    if iou_test > best_iou:
        best_iou = iou_test
        best_plan = env.plan
        best_tb = env.total_brick
    iou_test_total += iou_test
    reward_test_total += reward_test

reward_test_total = reward_test_total / N_iteration_test
iou_test_total = iou_test_total / N_iteration_test
secs = int(time.time() - start_time_test)
mins = secs // 60
secs = secs % 60
print(f"time = {mins} min {secs} sec")
print(f"iou = {iou_test_total}")
print(f"reward_test = {reward_test_total}")
env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)
iou_all_average += iou_test_total
iou_all_min = min(iou_min,iou_all_min)
save_path = "plots/"
plt.savefig(save_path+"PPO_Plan"+'.png')

iou_all_average = iou_all_average/10
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
