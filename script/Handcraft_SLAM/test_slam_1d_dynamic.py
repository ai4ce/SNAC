import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import os
from slam_1d import RobotMove
import statistics
sys.path.append('../../Env/1D/')
from DMP_Env_1D_dynamic_usedata_plan import deep_mobile_printing_1d1r
save_path = "./SLAM/plot/1D/Dynamic/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)
print('1D_Dynamic')
N_iteration_test = 500
iou_all_average = 0
iou_all_min = 1
iou_his=[]
env = deep_mobile_printing_1d1r(data_path="../../Env/1D/data_1d_dynamic_sin_envplan_500_test.pkl", random_choose_paln=False)
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
    state = env.reset()
    prev_state=state[0]
    env_plan=state[1]
    reward_test = 0
    robot_decider = RobotMove(env)
    while True:
        robot_decider.step()
        state_next, r, done = robot_decider.observation, robot_decider.reward, robot_decider.done
        reward_test += r
        if done:
            break
        prev_state = state_next[0]
    iou_test = env.iou()
    iou_min = min(iou_min, iou_test)
    iou_his.append(iou_test)
    if iou_test > best_iou:
        best_plan=env.plan
        best_iou = iou_test
        best_step = env.count_step
        best_brick = env.conut_brick
        best_tb = env.total_brick
        best_env = env.environment_memory
    iou_test_total += iou_test
    reward_test_total += reward_test
reward_test_total = reward_test_total / N_iteration_test
iou_test_total = iou_test_total / N_iteration_test
secs = int(time.time() - start_time_test)
mins = secs / 60
secs = secs % 60
env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test,best_env=best_env,
            best_iou=best_iou,best_step=best_step,best_brick=best_brick, best_plan=best_plan)
iou_all_average += iou_test_total
iou_all_min = min(iou_min,iou_all_min)
plt.savefig(save_path+"Plan_test_"+'.png')
iou_all_average = iou_all_average
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
std = statistics.pstdev(iou_his)
print("std",std)
