from agents.gcsl import GCSL
from agents.bc import BC
from agents.goplan import GOPlan
from agents.geaw import GEAW
from agents.wgcsl import WGCSL
from agents.td3bc import TD3BC
from agents.crl import CRL


def return_agent(**agent_params):
    if agent_params['agent'] == 'gcsl':
        return GCSL(**agent_params)
    elif agent_params['agent'] == 'bc':
        return BC(**agent_params)
    elif agent_params['agent'] == 'goplan':
        return GOPlan(**agent_params)
    elif agent_params['agent'] == 'geaw':
        return GEAW(**agent_params)
    elif agent_params['agent'] == 'wgcsl':
        return WGCSL(**agent_params)
    elif agent_params['agent'] == 'td3bc':
        return TD3BC(**agent_params)
    elif agent_params['agent'] == 'contrastive':
        return CRL(**agent_params)
    else:
        raise Exception('Invalid agent!')