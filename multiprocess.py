import sys
sys.path.append('./Env/1D')
sys.path.append('./Env/2D')
sys.path.append('./Env/3D')
from DMP_Env_1D_static import deep_mobile_printing_1d1r as deep_mobile_printing_1d1r_static
from DMP_Env_1D_dynamic_usedata_plan import deep_mobile_printing_1d1r as deep_mobile_printing_1d1r_dynamic
from DMP_Env_2D_static import deep_mobile_printing_2d1r as deep_mobile_printing_2d1r_static
from DMP_Env_2D_dynamic_usedata_plan import deep_mobile_printing_2d1r as deep_mobile_printing_2d1r_dynamic
from DMP_simulator_3d_static_circle import deep_mobile_printing_3d1r as deep_mobile_printing_3d1r_static
from DMP_simulator_3d_dynamic_triangle_usedata import deep_mobile_printing_3d1r as deep_mobile_printing_3d1r_dynamic
import gym
import numpy as np
import argparse

class VectorizedEnvWrapper(gym.Wrapper):
    def __init__(self, env_, num_envs=1):
        super().__init__(env_)
        self.num_envs = num_envs
        self.envs = [env_ for env_index in range(num_envs)]
    def reset(self):
        return np.asarray([env_item.reset() for env_item in self.envs])
    def reset_at(self, env_index):
        return np.asarray(self.envs[env_index].reset())
    def step(self, actions):
        next_states, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, done = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return np.asarray(next_states), np.asarray(rewards), \
            np.asarray(dones)

def main(args):
    num_envs = args.num_envs
    if args.env is None:
        print("please choose an environment in the list: {1DStatic, 1DDynamic,2DStatic, 2DDynamic, 3DStatic, 3DDynamic} ") 
        return
    elif args.env == "1DStatic":
        if args.plan_type is None:
            print("please choose a shape from list: {0: sin, 1:Gaussian, 2: Step}")
            return 
        else:
            env_input = deep_mobile_printing_1d1r_static(plan_choose=args.plan_type)
    elif args.env == "1DDynamic":
        env_input = deep_mobile_printing_1d1r_dynamic(data_path="./Env/1D/data_1d_dynamic_sin_envplan_500_train.pkl")

    elif args.env == "2DStatic":
        if args.plan_type is None:
            print("please choose a shape from list: {0: Dense, 1: Sparse}")
            return 
        else:
            env_input = deep_mobile_printing_2d1r_static(plan_choose=args.plan_type)
    elif args.env == "2DDynamic":
        if args.plan_type is None:
            print("please choose a shape from list: {0: Dense, 1: Sparse}")
            return 
        else:
            PLAN_LIST=["dense","sparse"]
            PLAN_NAME=PLAN_LIST[args.plan_type]
            env_input = deep_mobile_printing_2d1r_dynamic(data_path="./Env/2D/data_2d_dynamic_"+PLAN_NAME+"_envplan_500_train.pkl")

    elif args.env == "3DStatic":
        if args.plan_type is None:
            print("please choose a shape from list: {0: Dense, 1: Sparse}")
            return 
        else:
            env_input = deep_mobile_printing_3d1r_static(plan_choose=args.plan_type)
    elif args.env == "3DDynamic":
        if args.plan_type is None:
            print("please choose a shape from list: {0: Dense, 1: Sparse}")
            return 
        else:
            PLAN_LIST=["dense","sparse"]
            PLAN_NAME=PLAN_LIST[args.plan_type]
            env_input = deep_mobile_printing_3d1r_dynamic(data_path="./Env/2D/data_2d_dynamic_"+PLAN_NAME+"_envplan_500_train.pkl")

    env = VectorizedEnvWrapper(env_input, num_envs=num_envs)
    T = env.envs[0].total_step
    observations = env.reset()
    
    for t in range(T):
        actions = np.random.randint(3, size=num_envs)
        observations, rewards, dones = env.step(actions)  
    print(observations.shape)
    print(rewards.shape)
    print(dones.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=None,
                        help='Environment Name: {1DStatic, 1DDynamic,2DStatic, 2DDynamic, 3DStatic, 3DDynamic}')
    parser.add_argument('--plan_type', type=int, default=None,
                        help='type of shapes')
    parser.add_argument('--num_envs', type=int, default=3,
                        help='Number of environments')
    main(parser.parse_args())
    
