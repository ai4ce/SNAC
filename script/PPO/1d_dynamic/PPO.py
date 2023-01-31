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
import sys
sys.path.append('../../../')
import utils
import argparse

print("successful load module")
def main(cfg):
    config_path = cfg.config_path
    if config_path is None:
        print('Please provide a config path')
        return
    args = utils.read_config(config_path)  
    seeds=args.Random_seed
    OUT_FILE_NAME="PPO_1d_sin"+"_seed_"+str(seeds)
    log_path=args.model_dir
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    arg = {"gamma":args.gamma, "n_steps":args.n_steps, "noptepochs":args.noptepochs, "ent_coef":args.ent_coef,\
     "learning_rate":args.learning_rate, "vf_coef":args.vf_coef, "cliprange":args.cliprange, "nminibatches":args.nminibatches, \
     "tb_log_name":OUT_FILE_NAME,\
     "model_save_path":log_path+OUT_FILE_NAME ,\
     "tensorboard_log": args.log_dir+"/"+OUT_FILE_NAME ,\
     "seed":seeds}
    env = DMP_Env_1D_dynamic_usedata_plan.deep_mobile_printing_1d1r(data_path="./data_1d_dynamic_sin_envplan_500_train.pkl")
    env = make_vec_env(lambda: env, n_envs=1)
    policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[512, 512, 512])
    model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, gamma=arg["gamma"], n_steps=arg["n_steps"], noptepochs=arg["noptepochs"], ent_coef=arg["ent_coef"], learning_rate=arg["learning_rate"], vf_coef=arg["vf_coef"], cliprange=arg["cliprange"], nminibatches=arg["nminibatches"], verbose=1, tensorboard_log=arg["tensorboard_log"],n_cpu_tf_sess=1,seed=arg["seed"])
    time_steps = 1e7
    model.learn(total_timesteps=int(time_steps), tb_log_name=arg["tb_log_name"])
    model.save(arg["model_save_path"])
    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    main(parser.parse_args())
