from agents.gcsl import GCSL
from agents.bc import BC
from agents.goplan import GOPlan
from agents.geaw import GEAW
from agents.wgcsl import WGCSL
from agents.td3bc import TD3BC
from agents.crl import CRL


def return_agent(**agent_params):
    if agent_params['agent'] == 'GCSL':
        return GCSL(**agent_params)
    elif agent_params['agent'] == 'BC':
        return BC(**agent_params)
    elif agent_params['agent'].startswith('GOPlan'):
        return GOPlan(**agent_params)
    elif agent_params['agent'] == 'GEAW':
        return GEAW(**agent_params)
    elif agent_params['agent'] == 'WGCSL':
        return WGCSL(**agent_params)
    elif agent_params['agent'] == 'TD3BC':
        return TD3BC(**agent_params)
    elif agent_params['agent'] == 'CRL':
        return CRL(**agent_params)
    else:
        raise Exception('Invalid agent!')