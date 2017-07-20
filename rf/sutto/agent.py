class agent:
    def __init__(self, observation, reward):
        self.observation = observation
        self.reward = reward

    def act(self):
        self._action()

    # The method is overiden by specific agent
    def _action(self):
        pass
