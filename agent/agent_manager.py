# class for managing agents to play in one environment
# required for even single agent


class AgentManager:

    def __init__(self):
        pass

    def train(self, agents, game, epochs, agents_trains_params=None, visualizer=None):
        assert len(agents) == game.nb_players, 'Number of agents must be equal to number of players set in game'
        assert len(agents) == len(agents_trains_params), 'Train params must be provided for each agent'

        win_count = 0
        scores = []

        for epoch in range(epochs):
            game.reset()
            loss = 0

            while not game.game_is_over():
                # play and do stuff

                if visualizer:
                    state = game.get_state()
                    visualizer.visualize_state(state)

            # get info about game score
            if game.game_is_won():
                win_count += 1
            current_score = game.get_score()
            scores.append(current_score)

            # print epoch summary
            print('Training {} agent(s) to play {}, epoch {}/{}. Loss: {}, Current game score: {}, Total win count: {}'.
                  format(len(agents), game.name, loss, epoch, epochs, current_score, win_count))

        # print summary of train wins/scores
        avg_score = sum(scores) / epochs
        print('End of training. Epochs: {}, Average Score: {}, Total win count: {}.'.format(epochs,
                                                                                            avg_score, win_count))

    def play(self, agents, game, epochs, visualizer=None):
        pass

