from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import time
import os
import sys
from DMP_Env_2D_dynamic_usedata_plan import deep_mobile_printing_2d1r
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import pickle

def iou(environment_memory,environment_plan,HALF_WINDOW_SIZE,plan_height,plan_width):
    component1=environment_plan[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                       HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
    component2=environment_memory[HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_height,\
                       HALF_WINDOW_SIZE:HALF_WINDOW_SIZE+plan_width].astype(bool)
    overlap = component1*component2 # Logical AND
    union = component1 + component2 # Logical OR 
    IOU = overlap.sum()/float(union.sum())
    return IOU

def main_exp(arg):
    env = deep_mobile_printing_2d1r(data_path="./data_2d_dynamic_"+arg["PLAN_NAME"]+"_envplan_500_test.pkl", random_choose_paln=False)

    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[512, 512, 512])
    model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, gamma=arg["gamma"], n_steps=arg["n_steps"], noptepochs=arg["noptepochs"], ent_coef=arg["ent_coef"], learning_rate=arg["learning_rate"], vf_coef=arg["vf_coef"], cliprange=arg["cliprange"], nminibatches=arg["nminibatches"], verbose=1, tensorboard_log=arg["tensorboard_log"],n_cpu_tf_sess=1,seed=arg["seed"])
    model = PPO2.load(arg["model_save_path"])
    
    best_iou=0
    best_env = np.array([])
    iou_test_total=0
    reward_test_total=0
    iou_min = 1
    iou_history = []
    start_time_test = time.time()
    best_plan = np.array([])
    best_step = 0
    best_brick = 0

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(arg["N_iteration_test"]):
        obs = env.reset()
            
        reward_test=0
        print('Ep:',i)
        while True:
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            reward_test += rewards
            if done:
                break
        iou_test=iou(env.environment_memory,env.plan,env.HALF_WINDOW_SIZE,env.plan_height,env.plan_width)
        iou_history.append(iou_test)

        iou_min = min(iou_min,iou_test)
        if iou_test > best_iou:
            best_iou = iou_test
            best_plan = env.plan
            best_step = env.count_step
            best_brick = env.count_brick
            best_tb = env.total_brick
            best_env = env.environment_memory

        iou_test_total+=iou_test
        reward_test_total+=reward_test

    reward_test_total=reward_test_total/arg["N_iteration_test"]
    iou_test_total=iou_test_total/arg["N_iteration_test"]

    # iou_all_average += iou_test_total
    # iou_all_min = min(iou_min,iou_all_min)

    #print("total_brick:", env.total_brick)
    #print('iou_his:',iou_history)
    #print('iou_min:',min(iou_history))
    env.render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=arg["N_iteration_test"],best_env=best_env,best_iou=best_iou,best_step=best_step,best_brick=best_brick,best_plan=best_plan)
    plt.savefig(arg["graph_save_path"]+".png")
    with open(arg["graph_save_path"] + ".pkl",'wb') as f:
        pickle.dump(iou_history, f)
    print(iou_test_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-choose', default=1, type=int, help='dynamic has: dense 0, sparse 1')
    parser.add_argument('--seed', default=1, type=int, help='random seed from 1-4')
    parser.add_argument('--lr', default=0.00025, type=float, help='learning rate')

    args = parser.parse_args()
    PALN_CHOICE=args.plan_choose   # 0 dense, 1 sparse
    PLAN_LIST=["dense","sparse"]
    PLAN_NAME=PLAN_LIST[PALN_CHOICE]
    seeds=args.seed
    OUT_FILE_NAME="PPO_2d_"+PLAN_NAME+"_lr"+str(args.lr)+"_seed_"+str(seeds)+".zip"
    log_path="./PPO/2D/log/dynamic/"
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    arg = {"gamma":0.99, "n_steps":100000, "noptepochs":4, "ent_coef":0.01, \
    "learning_rate":args.lr, "vf_coef":0.5, "cliprange":0.1, "nminibatches":100, "N_iteration_test": 1, \
    "tb_log_name":OUT_FILE_NAME,\
    "model_save_path":log_path+OUT_FILE_NAME ,\
    "graph_save_path":log_path+"_ppt_plan"+PLAN_NAME+ "_seed"+str(seeds),\
    "tensorboard_log": "./PPO/2D/dynamic/",\
    "seed":seeds,\
    "plan_choose":PALN_CHOICE,\
     "PLAN_NAME": PLAN_NAME}
    test_agent = main_exp(arg)