env_name='FetchPush-v1'
dataset='datasets/push/expert/FetchPush/push_right_5000'
project='ago_OOD_push_new'

agent='td3bc'
gpu=0
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent} &

gpu=1
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent} &

gpu=2
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent} &

gpu=3
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent} &

gpu=0
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}



agent='wgcsl'
gpu=0
seed=0
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed}  --dataset ${dataset} --project ${project}  --group ${agent} &

gpu=1
seed=1
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent} &

gpu=2
seed=2
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent} &

gpu=3
seed=3
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent} &

gpu=0
seed=4
CUDA_VISIBLE_DEVICES=${gpu} python train.py --env_name ${env_name}  --agent ${agent} --seed ${seed} --dataset ${dataset} --project ${project}  --group ${agent}
