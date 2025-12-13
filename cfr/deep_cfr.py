import random
from game import StochasticGame

class DeepCFR:
    def __init__(self, game):
        self.game = game
        self.regret_sum = {} # R[I][a]
        self.strategy_sum = {} # s[I][a]
        self.strategy = {} # S[I][a]

    