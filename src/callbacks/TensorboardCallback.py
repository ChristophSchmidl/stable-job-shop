from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
 
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # print(self.locals["rewards"])
        # print(self.locals)
        global top_score
        score = self.locals["infos"][0]["score"]
        self.logger.record('a_score', score)
        if (top_score == None):
            top_score = score
            model.save("./model_DQN")
        elif (score > top_score):
            top_score = score
            model.save("./model_DQN")
        return True