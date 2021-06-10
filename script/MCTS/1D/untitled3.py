import sys
sys.path.append('../../../Env/1D/')
import os 

import gym
from DMP_Env_1D_static_MCTS import deep_mobile_printing_1d1r_MCTS
import matplotlib.pyplot as plt
import os
import uct
import numpy as np
import time 

save_path = "./log/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
### Parameters
env = deep_mobile_printing_1d1r_MCTS(plan_choose=2)
agent = uct.UCT(
    action_space=env.action_space,
    rollouts=10,
    horizon=2,
    is_model_dynamic=True
)

### Run
# =============================================================================
# env.reset()
# done = False
# 
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(1, 1, 1)
# print(env.total_brick)
# print(env.one_hot)
# ax.clear()
# Total_reward=0
# while True:
#     state, observation, reward, done = env.step(agent.act(env,done))
# # =============================================================================
# #     print(env.environment_memory)
# # =============================================================================
# # =============================================================================
# #     env.render(ax)
# #     env.iou()
# #     plt.pause(0.1)
# # =============================================================================
#     Total_reward+=reward
#     if done:
#         break
# # =============================================================================
# #     plt.show()
# # =============================================================================
# print(Total_reward)
# =============================================================================\
iou_all_average = 0
iou_all_min = 1
N_iteration_test=3
best_iou = 0
best_env = np.array([])
iou_test_total = 0
iou_min = 1
reward_test_total = 0
start_time_test = time.time()
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
for ep in range(N_iteration_test):
    print(ep)
    env.reset()
    reward_test = 0
    done=False
    while True:
        state, observation, reward, done = env.step(agent.act(env,done))
        reward_test += reward
        if done:
            break

    iou_test = env.iou()
    iou_min = min(iou_min, iou_test)

    if iou_test > best_iou:
        best_iou = iou_test
        best_plan = env.plan
        best_step = env.count_step
        best_brick = env.conut_brick
        best_env = env.environment_memory
        best_tb = env.total_brick
    iou_test_total += iou_test
    reward_test_total += reward_test

reward_test_total = reward_test_total / N_iteration_test
iou_test_total = iou_test_total / N_iteration_test
secs = int(time.time() - start_time_test)
mins = secs / 60
secs = secs % 60
env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick)

iou_all_average += iou_test_total
iou_all_min = min(iou_min,iou_all_min)
plt.savefig(save_path+"Plan"+str(2)+'.png')