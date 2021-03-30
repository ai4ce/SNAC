import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import os
from slam_1d_no_map import RobotMove

pth_plan = '7456'

sys.path.append('../../Env/1D/')
from DMP_Env_1D_dynamic_test import deep_mobile_printing_1d1r

save_path = "./log/SLAM/plot/1D/Dynamic/with_out_map/"
if os.path.exists(save_path) == False:
    os.makedirs(save_path)

print('1D_Dynamic')
iou_all_average = 0
iou_all_min = 1
for plan_choose in range(10):

    env = deep_mobile_printing_1d1r(plan_choose=plan_choose)
    ######################
    # hyper parameter
    minibatch_size = 64
    Lr = 0.00001
    N_iteration = 6000
    N_iteration_test = 200
    alpha = 0.9
    Replay_memory_size = 1000
    Update_traget_period = 200
    Action_dim = env.action_dim
    State_dim = env.state_dim
    hidden_state_dim = 256
    Time_step = 20
    UPDATE_FREQ = 5
    INITIAL_EPSILON = 0.1
    FINAL_EPSILON = 0.0
    ######################
    PLAN_DIM = env.plan_width
    env.reset()
    print("State_dimension:", State_dim)
    print("Total Brick: ", env.total_brick)
    print("plan_width", PLAN_DIM)
    print('######################')

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

        if iou_test > best_iou:
            best_iou = iou_test
            best_step = env.count_step
            best_brick = env.count_brick
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
               best_iou=best_iou,best_step=best_step,best_brick=best_brick)
    iou_all_average += iou_test_total
    iou_all_min = min(iou_min,iou_all_min)
    plt.savefig(save_path+"Plan_test_"+str(plan_choose)+'.png')

iou_all_average = iou_all_average/10
print('iou_all_average',iou_all_average)
print('iou_all_min',iou_all_min)
plt.show()

#with map
# iou_all_average 0.9443759847945794
# iou_all_min 0.43

#no_map
# iou_all_average 0.7082887240186653
# iou_all_min 0.41294117647058826