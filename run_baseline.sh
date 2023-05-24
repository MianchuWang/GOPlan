env_name='FetchPush-v1'
dataset='datasets/push/expert/FetchPush/push_right_5000'
project='ago_OOD_push_new'


agent='wgcsl'
gpu=0
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000  &

gpu=1
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=2
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=0
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &




agent='gcsl'
gpu=1
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=2
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=1
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=0
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000



agent='geaw'
gpu=2
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=1
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=2
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=0
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &


agent='td3bc'
gpu=0
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=1
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=2
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000





env_name='FetchPickAndPlace-v1'
dataset='datasets/pick/expert/FetchPick/pick_right_5000'
project='ago_OOD_pick_new'


agent='wgcsl'
gpu=0
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000  &

gpu=1
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=2
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=0
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &




agent='gcsl'
gpu=1
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=2
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=1
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=0
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000



agent='geaw'
gpu=2
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=1
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=2
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=0
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &


agent='td3bc'
gpu=0
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=1
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=2
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000 &

gpu=3
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 260000


