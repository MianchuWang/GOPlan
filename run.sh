env_name='FetchPush-v1'
dataset='datasets/push/expert/FetchPush/push_right_5000'
project='ago_OOD_push_new'

# dataset='datasets/cross/cross_merge_10_new'
# env_name='PointCross'

agent='ago'
gpu=7
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &

gpu=8
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &

gpu=8
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &

gpu=9
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &

gpu=9
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &


env_name='FetchPickAndPlace-v1'
dataset='datasets/pick/expert/FetchPick/pick_right_5000'
project='ago_OOD_pick_new'


gpu=7
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &

gpu=6
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &

gpu=6
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &

gpu=5
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000 &

gpu=5
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}_new --pretrain_steps 250000
