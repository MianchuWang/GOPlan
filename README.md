


# GOPlan: Goal-conditioned Offline Reinforcement Learning by Planning with Learned Models

This is the code for the paper *GOPlan: Goal-conditioned Offline Reinforcement Learning by Planning with Learned Models*.  In this work, we propose a novel two-stage model-based framework, *Goal-conditioned Offline Planning (GOPlan)*, including (1) pretraining a prior policy capable of capturing multi-modal action distribution within the multi-goal dataset; (2) employing the reanalysis method with planning to generate imagined trajectories for funetuning policies. Through experimental evaluations, we demonstrate that GOPlan achieves state-of-the-art performance on various offline multi-goal manipulation and navigation tasks. Moreover, our results highlight the superior ability of GOPlan to handle small data budgets and generalize to OOD goals.

## Requirements
* python = 3.9.0
* d4rl = 1.1
* gym = 0.26.2
* gym-robotics = 1.2.2
* mujoco-py = 2.1.2.14
* pytorch = 1.13.1

## Supported Environments
**Gym Robotics**:  FetchReach-v1, FetchPush-v1, FetchPickAndPlace-v1, FetchSlide-v1, HandReach-v0.
**OOD Robotics**: FetchPushOOD-Right2Right-v1, FetchPushOOD-Left2Left-v1, FetchPushOOD-Left2Right-v1, FetchPushOOD-Right2Left-v1, FetchPickOOD-Right2Right-v1, FetchPickOOD-Right2Left-v1, FetchPickOOD-Left2Left-v1, FetchPickOOD-Left2Right-v1.
**Others**: PointRooms, PointReach, SawyerReach, SawyerDoor, Reacher-v2.

More details can be found in ```GOPlan/envs/__init__.py```.

## Usage
### Preparation
* Clone the repo and change the directory:
  ```
  $ git clone https://github.com/MianchuWang/GOPlan.git
  $ cd GOPlan
  ```
* The offline benchmark used in the paper is introduced by [*Yang et al.* ](https://arxiv.org/abs/2202.04478)  Before running the code, please download the ``buffer.pkl`` files from [Google Drive](https://drive.google.com/drive/folders/1SIo3qFmMndz2DAnUpnCozP8CpG420ANb) as the offline datasets.
* Put the datasets in the specified folder. For example, when you download from Google Drive ``offline_data/expert/FetchReach/buffer.pkl``, you shall put the file into the folder ``GOPlan/datasets/`` and rename it as ``FetchReach.pkl``.
### Run algorithms on the benchmark
* Run the following code to train GOPlan agent on the FetchReach dataset: 
  ```
  $ python train.py --agent goplan --env_name FetchReach-v1 --dataset GOPlan/datasets/FetchReach
  ```
* Run the following code to train [GCSL](https://arxiv.org/abs/1912.06088) agent on the FetchReach dataset:
  ```
  $ python train.py --agent gcsl --env_name FetchReach-v0 --dataset GOPlan/datasets/FetchReach
  ```
  * Run the following code to train [WGCSL](https://arxiv.org/abs/2202.04478) agent on the FetchReach dataset:
  ```
  $ python train.py --agent wgcsl --env_name FetchReach-v0 --dataset GOPlan/datasets/FetchReach
  ```
  The code also supports [TD3BC](https://arxiv.org/abs/2106.06860), [contrastive RL](https://arxiv.org/abs/2206.07568), etc. More details can be found in ```GOPlan/agents/__init__.py```
  
## Code References

1. Rui Yang, Yiming Lu, Wenzhe Li, Hao Sun, Meng Fang, Yali Du, Xiu Li, Lei Han, & Chongjie Zhang (2022). Rethinking Goal-Conditioned Supervised Learning and Its Connection to Offline RL. In *International Conference on Learning Representations*. [[code]](https://github.com/YangRui2015/AWGCSL)
2. Rui Yang, Lin Yong, Xiaoteng Ma, Hao Hu, Chongjie Zhang, & Tong Zhang (2023). What is Essential for Unseen Goal Generalization of Offline Goal-conditioned RL?. In *International Conference on Machine Learning*. [[code]](https://github.com/YangRui2015/GOAT)
3. Henry Charlesworth, Giovanni Montana (2020). PlanGAN: Model-based Planning With Sparse Rewards and Multiple Goals. In *Advances in Neural Information Processing Systems*. [[code]](https://github.com/henrycharlesworth/PlanGAN)
