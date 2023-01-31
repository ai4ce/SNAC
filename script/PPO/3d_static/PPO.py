from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from DMP_simulator_3d_static_circle import deep_mobile_printing_3d1r
import tensorflow as tf
import os
import argparse
import sys
sys.path.append('../../../')
import utils

def main(cfg):
    config_path = cfg.config_path
    if config_path is None:
        print('Please provide a config path')
        exit()
    args = utils.read_config(config_path)

    PALN_CHOICE=args.plan_type   # 0 dense, 1 sparse
    PLAN_LIST=["dense","sparse"]
    PLAN_NAME=PLAN_LIST[PALN_CHOICE]
    seeds=args.Random_seed
    OUT_FILE_NAME="PPO_3d_"+PLAN_NAME+"_seed_"+str(seeds)
    log_path = args.model_dir
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    arg = {"gamma":args.gamma, "n_steps":args.n_steps, "noptepochs":args.noptepochs, "ent_coef":args.ent_coef, \
    "learning_rate":args.learning_rate, "vf_coef":args.vf_coef, "cliprange":args.cliprange, "nminibatches":args.nminibatches, \
    "tb_log_name":OUT_FILE_NAME,\
    "model_save_path":log_path+"/"+OUT_FILE_NAME ,\
    "tensorboard_log": args.log_dir,\
    "seed":seeds,\
    "plan_choose":PALN_CHOICE}

    env = deep_mobile_printing_3d1r(plan_choose=arg["plan_choose"])
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