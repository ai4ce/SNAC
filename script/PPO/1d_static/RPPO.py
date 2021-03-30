from stable_baselines.common.policies import LstmPolicy, MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

import time
from DMP_Env_1D_static import deep_mobile_printing_1d1r

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import tensorflow as tf




plan = 0
def main_exp(arg):
    env = deep_mobile_printing_1d1r(plan_choose=plan)
    env = make_vec_env(lambda: env, n_envs=20)

    class CustomLSTMPolicy(LstmPolicy):
        def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
            super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                net_arch=[128, 'lstm', dict(vf=[10], pi=[10])],
                                layer_norm=True, feature_extraction="mlp", **_kwargs)

    model = PPO2(CustomLSTMPolicy, env, gamma=arg["gamma"], n_steps=arg["n_steps"], noptepochs=arg["noptepochs"], ent_coef=arg["ent_coef"], learning_rate=arg["learning_rate"], vf_coef=arg["vf_coef"], cliprange=arg["cliprange"], nminibatches=arg["nminibatches"], verbose=1, tensorboard_log="./log_ppo/")
    time_steps = 1e6
    model.learn(total_timesteps=int(time_steps), tb_log_name=arg["tb_log_name"])
    return model



if __name__ == "__main__":
    arg = {"gamma":0.99, "n_steps":1000, "noptepochs":4, "ent_coef":0.01, "learning_rate":0.00025, "vf_coef":0.5, "cliprange":0.1, "nminibatches":20, "tb_log_name":"RPPO"}
    model = main_exp(arg)

    print("Testing")

    # Retrieve the env
    env = model.get_env()
    test_agent = model
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
   
        state = None 
        done = [False for _ in range(env.num_envs)]
        while True:
            # We need to pass the previous state and a mask for recurrent policies
            # to reset lstm state when a new episode begin
            if ep == N_iteration_test -1:
                env.envs[0].render(ax,iou_average=iou_test_total,iou_min=iou_min,iter_times=N_iteration_test)
            action, state = model.predict(obs, state=state, mask=done)
            iou_test = env.envs[0].iou()                                     ##### NEED TO FIND ANOTHER WAY : does not count last brick...
            obs, r, done, _ = env.step(action)
            reward_test += r
            # Note: with VecEnv, env.reset() is automatically called
            if done[0]:
                break

        iou_min = min(iou_min, iou_test)

        if iou_test > best_iou:
            best_iou = iou_test

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

    save_path = "plots/"
    plt.savefig(save_path+"RPPO_Plan"+str(plan)+'.png')
