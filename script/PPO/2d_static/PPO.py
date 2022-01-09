from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import time
from DMP_Env_2D_static import deep_mobile_printing_2d1r
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf
import os
print("successful load module")


def main_exp(arg):
    env = deep_mobile_printing_2d1r(plan_choose=arg["plan_choose"])
    env = make_vec_env(lambda: env, n_envs=1)
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[512, 512, 512])
    model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, gamma=arg["gamma"], n_steps=arg["n_steps"], noptepochs=arg["noptepochs"], ent_coef=arg["ent_coef"], learning_rate=arg["learning_rate"], vf_coef=arg["vf_coef"], cliprange=arg["cliprange"], nminibatches=arg["nminibatches"], verbose=1, tensorboard_log=arg["tensorboard_log"],n_cpu_tf_sess=1,seed=arg["seed"])
    time_steps = 1e7
    model.learn(total_timesteps=int(time_steps), tb_log_name=arg["tb_log_name"])
    model.save(arg["model_save_path"])
    return model


if __name__ == "__main__":
    PALN_CHOICE=1   # 0 dense, 1 sparse
    PLAN_LIST=["dense","sparse"]
    PLAN_NAME=PLAN_LIST[PALN_CHOICE]
    seeds=5
    OUT_FILE_NAME="PPO_2d_"+PLAN_NAME+"_seed_"+str(seeds)
    log_path="/mnt/NAS/home/WenyuHan/SNAC/PPO/2D/log/static/"
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    arg = {"gamma":0.99, "n_steps":100000, "noptepochs":4, "ent_coef":0.01, \
    "learning_rate":0.00025, "vf_coef":0.5, "cliprange":0.1, "nminibatches":100, \
    "tb_log_name":OUT_FILE_NAME,\
    "model_save_path":log_path+OUT_FILE_NAME ,\
     "tensorboard_log": "/mnt/NAS/home/WenyuHan/SNAC/PPO/2D/static/",\
     "seed":seeds,\
     "plan_choose":PALN_CHOICE}
    test_agent = main_exp(arg)

    # print(f"Testing plan {plan}")

    # env = deep_mobile_printing_2d1r(plan_choose=plan)


    # def iou(environment_memory,environment_plan,HALF_WINDOW_SIZE,plan_height,plan_width):
    #     component1=environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
    #                        HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
    #     component2=environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
    #                        HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
    #     overlap = component1*component2 # Logical AND
    #     union = component1 + component2 # Logical OR
    #     IOU = overlap.sum()/float(union.sum())
    #     return IOU


    # N_iteration_test = 500
    # best_iou = 0
    # iou_test_total = 0
    # iou_min = 1
    # reward_test_total = 0
    # start_time_test = time.time()

    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(1, 1, 1)
    # for ep in range(N_iteration_test):
    #     obs = env.reset()
    #     reward_test = 0

    #     while True:
    #         action, _ = test_agent.predict(obs)
    #         obs, r, done, info = env.step(action)
    #         reward_test += r
    #         if done:
    #             break

    #     iou_test = iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
    #     iou_min = min(iou_min, iou_test)

    #     if iou_test > best_iou:
    #         best_iou = iou_test
    #         best_plan = env.plan
    #         best_tb = env.total_brick
    #     iou_test_total += iou_test
    #     reward_test_total += reward_test

    # reward_test_total = reward_test_total / N_iteration_test
    # iou_test_total = iou_test_total / N_iteration_test
    # secs = int(time.time() - start_time_test)
    # mins = secs // 60
    # secs = secs % 60
    # print(f"time = {mins} min {secs} sec")
    # print(f"iou = {iou_test_total}")
    # print(f"reward_test = {reward_test_total}")
    # env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)

    # save_path = "plots/"
    # plt.savefig(save_path+"PPO_Plan"+str(plan)+'.png')
