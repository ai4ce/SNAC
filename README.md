# Simultaneous Navigation and Construction Benchmarking Environments

[**Wenyu Han**](https://github.com/WenyuHan-LiNa), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng), [**Haoran Wu**](https://www.linkedin.com/in/haoran-lucas-ng-4053471a0/), [**Alexander Gao**](https://www.alexandergao.com/), [**Armand Jordana**](https://wp.nyu.edu/machinesinmotion/people/), [**Dong Liu**](http://mechatronics.engineering.nyu.edu/people/phd-candidates/dongdong-liu.php), [**Lerrel Pinto**](https://www.lerrelpinto.com/), [**Ludovic Righetti**](https://wp.nyu.edu/machinesinmotion/89-2/)

## Abstract
We need intelligent robots for mobile construction, the process of navigating in an environment and modifying its structure according to a geometric design. In this task, a major robot vision and learning challenge is how to exactly achieve the design without GPS, due to the difficulty caused bythe bi-directional coupling of accurate robot localization and navigation together with strategic environment manipulation. However, many existing robot vision and learning tasks such as visual navigation and robot manipulation address only one of these two coupled aspects. To stimulate the pursuit of a generic and adaptive solution, we reasonably simplify mobile construction as a partially observable Markov decision process (POMDP) in 1/2/3D grid worlds and benchmark the performance of a handcrafted policy with basic localization and planning, and state-of-the-art deep reinforcement learning (RL) methods. Our extensive experiments show that the coupling makes this problem very challenging for those methods, and emphasize the need for novel task-specific solutions.

## [Code (GitHub)](https://github.com/ai4ce/SNAC) & Dependencies
All environment scripts can be found in [Env](https://github.com/ai4ce/SNAC/tree/main/Env) folder. These environments are developed based on the [OpenAi Gym](https://gym.openai.com/). All baseline scripts are in [script](https://github.com/ai4ce/SNAC/tree/main/script) floder. You need to install the [Pytorch](https://pytorch.org/) to run all baseline scripts. We use [Stable baseline](https://github.com/openai/baselines/) for PPO algorithm. 
## [How to use]

Our environment is developed based on the [OpenAi Gym](https://gym.openai.com/). You can simply follow the similar way to use our environment. Here we present a example for using 1D static task environment.
```
from DMP_Env_1D_static import deep_mobile_printing_1d1r ### you may need to find the path to this environment 
env = deep_mobile_printing_1d1r(plan_choose=2) ### plan_choose could be 0: sin, 1: Gaussian, and 2: Step curve  
observation = env.reset()
for _ in range(1000):
  action = np.random.randint(env.action_dim) # your agent here (this takes random actions)
  observation, reward, done = env.step(action)
  if done:
    break
```

## [Paper (arXiv)](https://arxiv.org/abs/2103.16732)
To cite our paper:
```
@misc{han2021simultaneous,
      title={Simultaneous Navigation and Construction Benchmarking Environments}, 
      author={Wenyu Han and Chen Feng and Haoran Wu and Alexander Gao and Armand Jordana and Dong Liu and Lerrel Pinto and Ludovic Righetti},
      year={2021},
      eprint={2103.16732},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Acknowledgment
 The research is supported by NSF CPS program under CMMI-1932187. The authors gratefully thank our human test participants and the helpful comments from [**Bolei Zhou**](http://bzhou.ie.cuhk.edu.hk/), [**Zhen Liu**](http://itszhen.com/), and the anonymous reviewers, and also [**Congcong Wen**](https://scholar.google.com/citations?hl=en&user=OTBgvCYAAAAJ) for paper revision.
