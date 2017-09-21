from .agent import Agent


class DQNAgent(Agent):

    def __init__(self, model, memory_size): # TODO memory/experience replay
        Agent.__init__(self, model)

    def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1, 0.1], epsilon_rate=0.5,
              visualizer=None, reset_memory=False, save_model=False):
        raise NotImplemented('Write me pls')

    def play(self, game, episodes, epsilon, visualizer=None):
        win_count = 0
        # memory?


        for episode in range(episodes):
            # restoration/initialization of initial environment state
            game.reset()
            game_over = False

            while not game_over:
                # do stuff


                # visualization of current state
                if visualizer:
                    pass
