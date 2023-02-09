# Learning Representations via a Robust Behavioral Metric for Deep Reinforcement Learning
This repository contains the code for the paper "Learning Representations via a Robust Behavioral Metric for Deep Reinforcement Learning" (RAP). If use this code, please cite our paper:
```
@inproceedings{
chen2022learning,
title={Learning Representations via a Robust Behavioral Metric for Deep Reinforcement Learning},
author={Jianda Chen and Sinno Pan},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=7YXXt9lRls}
}
```
# Clone This Repo
```
git clone --recurse-submodules https://github.com/jianda-chen/RAP_distance.git
```
Or
```
git clone https://github.com/jianda-chen/RAP_distance.git
cd RAP_distance
git submodule update --init --recursive
```

# Dependencies
* Python >= 3.8
* PyTorch >= 1.7
* DeepMind Control Suite. Please refer to [dm_control](https://github.com/deepmind/dm_control) page to install the dependencies for DeepMind Control Suite
* Other python packages. Install other python dependencies in conda environment by the following command:
```
conda env update -f conda_env.yaml
``` 
# Getting Started
To run the Distracting DeepMind Control Suite example, simply run the following command:
```
./run_local.sh
```
This script will save the experiment records in ```../log``` directory.

If you need to change the saving directory or run other tasks, please revise the file ```run_local.sh``` accordingly.


## License
This work is under [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
