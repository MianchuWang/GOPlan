env_name='FetchPush-v1'
cuda=0

agent='GOPlan-2'
size=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} 

agent='GOPlan-5'
size=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} 

agent='GOPlan-10'
size=10
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} 


env_name='FetchPickAndPlace-v1'

agent='GOPlan-2'
size=2
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} 

agent='GOPlan-5'
size=5
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} 

agent='GOPlan-10'
size=10
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} &
CUDA_VISIBLE_DEVICES=${cuda} python train.py --agent ${agent} --ensemble_size ${size} 
