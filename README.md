# Learning Simultaneous Navigation and Construction in Grid Worlds

[**Wenyu Han**](https://www.linkedin.com/in/wenyuhan0616), [**Haoran Wu**](https://www.linkedin.com/in/haoran-lucas-ng-4053471a0/), [**Eisuke Hirota**](https://www.linkedin.com/in/eisukeh/), [**Alexander Gao**](https://www.alexandergao.com/), [**Lerrel Pinto**](https://www.lerrelpinto.com/), [**Ludovic Righetti**](https://wp.nyu.edu/machinesinmotion/89-2/), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng), 

## Abstract
We propose to study a new learning task, mobile construction, to enable an agent to build designed structures in 1/2/3D grid worlds while navigating in the same evolving environments. Unlike existing robot learning tasks such as visual navigation and object manipulation, this task is challenging because of the interdependence between accurate localization and strategic construction planning. In pursuit of generic and adaptive solutions to this partially observable Markov decision process (POMDP) based on deep reinforcement learning (RL), we design a Deep Recurrent Q-Network (DRQN) with explicit recurrent position estimation in this dynamic grid world. Our extensive experiments show that pre-training this position estimation module before Q-learning can significantly improve the construction performance measured by the intersection-over-union score, achieving the best results in our benchmark of various baselines including model-free and model-based RL, a handcrafted SLAM-based policy, and human players.

# Installation

We recommend user to create a virtual environment for running this project. We list details of the environment setup process as follows:

## Create and activate new conda env.

```
conda create -n my-conda-env python=3.7
conda activate my-conda-env
```

## Note: pytorch is needed, so you need to install it based on your own system conditions. Here we use Linux and CUDA version 11.7 as an example.

```pip3 install torch torchvision torchaudio```

## Next, install other dependencies listed in requirement.txt

```pip install -r requirements.txt```

## How to use

Our environment is developed based on the [OpenAi Gym](https://gym.openai.com/). You can simply follow the similar way to use our environment. Here we present an example for using 1D static task environment.

```
from DMP_Env_1D_static import deep_mobile_printing_1d1r ### you may need to find the path to this environment in [Env] folder 
env = deep_mobile_printing_1d1r(plan_choose=2) ### plan_choose could be 0: sin, 1: Gaussian, and 2: Step curve  
observation = env.reset()
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.clear()
for _ in range(1000):
  action = np.random.randint(env.action_dim) # your agent here (this takes random actions)
  observation, reward, done = env.step(action)
  env.render(ax)
  plt.pause(0.1)
  if done:
    break
plt.show()
```

# Reproduce experiment results

All scripts for each method are in script/ folder where subfolder contains policies for 1D, 2D, and 3D tasks. You can find all hyperparameters used for each case in the config/ folder which has the same structure as script/ folder. The scripts for simulation environments are in Env/ folder. You can easily reproduce the experiments by running the algorithm scripts with its corresponding hyperparameters in the YML files. For example, if I want to train the DQN policy on 2D variable dense task:

```
cd script/DQN/2d/
python DQN_2d_dynamic.py ../../../config/DQN/2D/dynamic_dense.yml
```

# Multiprocess

We also provide a multiprocess script for batch simulation. 

```
python multiprocess.py --env 1DStatic --plan_type 0 --num_envs 5
```

## [Paper (OpenReview)](https://openreview.net/forum?id=NEtep2C7yD)
To cite our paper:
```
@inproceedings{
    anonymous2023learning,
    title={Learning Simultaneous Navigation and Construction in Grid Worlds},
    author={Anonymous},
    booktitle={Submitted to The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=NEtep2C7yD},
    note={under review}
}
```

## Acknowledgment
 The research is supported by NSF CPS program under CMMI-1932187. The authors gratefully thank our human test participants and the helpful comments from [**Bolei Zhou**](http://bzhou.ie.cuhk.edu.hk/), [**Zhen Liu**](http://itszhen.com/), and the anonymous reviewers, and also [**Congcong Wen**](https://scholar.google.com/citations?hl=en&user=OTBgvCYAAAAJ) for paper revision.
