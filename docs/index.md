# Simultaneous Navigation and Construction Benchmarking Environments

[**Wenyu Han**](https://github.com/WenyuHan-LiNa), [**Chen Feng**](https://engineering.nyu.edu/faculty/chen-feng), [**Haoran Wu**](), [**Alexander Gao**](https://www.alexandergao.com/), [**Armand Jordana**](https://wp.nyu.edu/machinesinmotion/people/), [**Dong Liu**](http://mechatronics.engineering.nyu.edu/people/phd-candidates/dongdong-liu.php), [**Lerrel Pinto**](https://www.lerrelpinto.com/), [**Ludovic Righetti**](https://wp.nyu.edu/machinesinmotion/89-2/)

![Overview](https://github.com/ai4ce/SNAC/blob/main/docs/figs/overview.PNG)

|[Abstract](#abstract)|[Code](#code-github)|[Paper](#paper-arxiv)|[Results](#results)|[Acknowledgment](#acknowledgment)|

## Abstract
We need intelligent robots for mobile construc-tion, the process of navigating in an environment and modifyingits structure according to a geometric design. In this task, a major robot vision and learning challenge is how to exactlyachieve the design without GPS, due to the difficulty caused bythe bi-directional coupling of accurate robot localization and navigation together with strategic environment manipulation. However, many existing robot vision and learning tasks such as visual navigation and robot manipulation address only one of these two coupled aspects. To stimulate the pursuit of a generic and adaptive solution, we reasonably simplify mobile construction as a partially observable Markov decision process (POMDP) in 1/2/3D grid worlds and benchmark the performance of a handcrafted policy with basic localization and planning, and state-of-the-art deep reinforcement learning (RL) methods. Our extensive experiments show that the coupling makes this problem very challenging for those methods, and emphasize the need for novel task-specific solutions.

## [Code (GitHub)](https://github.com/ai4ce/SNAC/tree/main/script)
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
"Simultaneous Navigation and Construction Benchmarking Environments". 
``` 
## [Paper (arXiv)]
To cite our paper:

### Task environment setups  
![env](https://github.com/ai4ce/SNAC/blob/main/docs/figs/environment.PNG)

## Comparison 
**Comparison between other robotic learning tasks with ours.**
|   | Loc |Plan|Env-Mod|Env-Eval|
| ------------- | ------------- |------------- |------------- |------------- |
| Robot Manipulation |:x:| :heavy_check_mark:  |:heavy_check_mark: |:x:|
| Robot Locomotion  | :x: |:x: |:x: |:x: |
|Visual Navigation|:heavy_check_mark:  |:heavy_check_mark: |:x:|:x:|
|Atari|:x:|:heavy_check_mark: / :x:|:heavy_check_mark: |:x:|
|Minecraft|:x:|:heavy_check_mark: / :x:|:heavy_check_mark: |:x:|
|First-Person-Shooting|:heavy_check_mark:|:x:|:x:|:x:|
|Real-Time Strategy Games|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:x:|
|Physical Reasoning|:x:|:heavy_check_mark:|:heavy_check_mark:|:x:|
|Mobile Construction (Ours)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|

## Results
**Benchmark results for all baselines, including human baseline: average IoU(left) and minimum IoU(right). Human data of 3D environment is not collected, because it is time-consuming for human to play one game.**
![Baseline_curve](https://github.com/ai4ce/SNAC/blob/main/docs/figs/result_curve.PNG)

**The best testing visualized results of baselines on all tasks.**
![Baseline_visualize](https://github.com/ai4ce/SNAC/blob/main/docs/figs/results_fig.PNG)

## Acknowledgment
 The research is supported by NSF CPS program under CMMI-1932187. The authors gratefully thank our human test participants and the helpful comments from [**Bolei Zhou**](http://bzhou.ie.cuhk.edu.hk/), [**Zhen Liu**](http://itszhen.com/), and the anonymous reviewers, and also [**Congcong Wen**](https://scholar.google.com/citations?hl=en&user=OTBgvCYAAAAJ) for paper revision.
