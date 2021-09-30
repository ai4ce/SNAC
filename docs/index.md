# Simultaneous Navigation and Construction Benchmarking Environments

[**Wenyu Han**](https://github.com/WenyuHan-LiNa), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng), [**Haoran Wu**](https://www.linkedin.com/in/haoran-lucas-ng-4053471a0/), [**Alexander Gao**](https://www.alexandergao.com/), [**Armand Jordana**](https://wp.nyu.edu/machinesinmotion/people/), [**Dong Liu**](http://mechatronics.engineering.nyu.edu/people/phd-candidates/dongdong-liu.php), [**Lerrel Pinto**](https://www.lerrelpinto.com/), [**Ludovic Righetti**](https://wp.nyu.edu/machinesinmotion/89-2/)

![Overview](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/overview.PNG?token=ANKETMQES4EKYCQYAE4K4WLAN4WB4)

|[Abstract](#abstract)|[Code](#code-github)|[Paper](#paper-arxiv)|[Results](#results)|[Acknowledgment](#acknowledgment)|

## Abstract
We need intelligent robots for mobile construction, the process of navigating in an environment and modifying its structure according to a geometric design. In this task, a major robot vision and learning challenge is how to exactly achieve the design without GPS, due to the difficulty caused bythe bi-directional coupling of accurate robot localization and navigation together with strategic environment manipulation. However, many existing robot vision and learning tasks such as visual navigation and robot manipulation address only one of these two coupled aspects. To stimulate the pursuit of a generic and adaptive solution, we reasonably simplify mobile construction as a partially observable Markov decision process (POMDP) in 1/2/3D grid worlds and benchmark the performance of a handcrafted policy with basic localization and planning, and state-of-the-art deep reinforcement learning (RL) methods. Our extensive experiments show that the coupling makes this problem very challenging for those methods, and emphasize the need for novel task-specific solutions.

## [Code (GitHub)](https://github.com/ai4ce/SNAC/tree/main/)
```
The code is copyrighted by the authors. Permission to copy and use 
 this software for noncommercial use is hereby granted provided: (a)
 this notice is retained in all copies, (2) the publication describing
 the method (indicated below) is clearly cited, and (3) the
 distribution from which the code was obtained is clearly cited. For
 all other uses, please contact the authors.
 
 The software code is provided "as is" with ABSOLUTELY NO WARRANTY
 expressed or implied. Use at your own risk.

This code provides an implementation of the method described in the
following publication: 

Wenyu Han, Chen Feng, Haoran Wu, Alexander Gao, Armand Jordana, Dong Liu, Lerrel Pinto, and Ludovic Righetti,    
"Simultaneous Navigation and Construction Benchmarking Environments (arXiv)". 
``` 
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

### Task environment setups  
![env](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/environment.PNG?token=ANKETMWZOL7HPJVJHNDX2B3AN4WDE)

## Comparison 
**Comparison between other robotic learning tasks with ours.**
![table](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/comparison_table.PNG?token=ANKETMXBFQCPNT5CBK5GVKDAN4YP4)

## Results
**Benchmark results for all baselines, including human baseline: average IoU(left) and minimum IoU(right). Human data of 3D environment is not collected, because it is time-consuming for human to play one game.**
![Baseline_curve](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/result_curve.PNG?token=ANKETMSFKVHG2SIV2JIVMLTAN4WEW)

**The best testing visualized results of baselines on all tasks.**
![Baseline_visualize](https://raw.githubusercontent.com/ai4ce/SNAC/main/docs/figs/results_fig.PNG?token=ANKETMRREVGARACAVL44QJLAN4WFW)

## Acknowledgment
 The research is supported by NSF CPS program under CMMI-1932187. The authors gratefully thank our human test participants and the helpful comments from [**Bolei Zhou**](http://bzhou.ie.cuhk.edu.hk/), [**Zhen Liu**](http://itszhen.com/), and the anonymous reviewers, and also [**Congcong Wen**](https://scholar.google.com/citations?hl=en&user=OTBgvCYAAAAJ) for paper revision.
