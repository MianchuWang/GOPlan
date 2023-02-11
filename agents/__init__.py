from agents.gcsl import GCSL
from agents.bc import BC


def return_agent(**agent_params):
    if agent_params['agent'] == 'gcsl':
        return GCSL(**agent_params)
    elif agent_params['agent'] == 'bc':
        return BC(**agent_params)
    elif agent_params['agent'] == 'ago':
        return AGO(**agent_params)
    else:
        raise Exception('Invalid agent!')