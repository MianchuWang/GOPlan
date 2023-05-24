# GOPlan: Offline Goal-conditioned Reinforcement Learning by Planning with Learned Models

This is the code for the paper *GOPlan: Offline Goal-conditioned Reinforcement Learning by Planning with Learned Models*.

## Usage:
* Download the offline datasets from [this](https://drive.google.com/drive/folders/1SIo3qFmMndz2DAnUpnCozP8CpG420ANb) anonymous Google Drive link. Further information can be found [here](https://github.com/YangRui2015/AWGCSL).
* Put the downded `.pkl` file in an appropriate folder. For example, put `FetchPickAndPlace.pkl` into folder `GOPlan/datasets/gym`.
* Run the following code to train GOPlan on FetchPickAndPlace dataset: 
  ```
  python train.py --env_name FetchPickAndPlace-v1 --dataset GOPlan/datasets/gym/FetchPickAndPlace
  ```
  
