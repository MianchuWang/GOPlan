from agents.bc import BC

class GCSL(BC):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.sample_func = self.replay_buffer.sample_her