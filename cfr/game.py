import random

def get_combos(target, num_slots, current_combo=None):
    if current_combo is None:
        current_combo = []
    if num_slots == 1:
        if target >= 0:
            yield tuple(current_combo + [target])
        return
    for i in range(target + 1):
        yield from get_combos(target - i, num_slots - 1, current_combo + [i])

class GameState:
    def __init__(self, s1, s2, history=None):
        self.s1 = s1
        self.s2 = s2
        self.history = history or []

    def infoset_key(self, player):
        # Player sees only their own local state + their own actions (i.e their state history)
        player_s = self.s1 if player == 1 else self.s2
        player_history = tuple(s[player-1] for s in self.history)
        return (player, player_s, player_history)

class StochasticGame:
    @classmethod
    def generate_states(cls, p, q):
        return tuple({combo for tot in range(p+1) for combo in get_combos(tot, q)})

    @classmethod
    def get_actions_for(cls, s):
        p, q = sum(s), len(s)
        new_states = StochasticGame.generate_states(p, q)
        return {
            tuple(new_s[i] - s[i] for i in range(q))
            for new_s in new_states
        }

    def __init__(self, p1, p2, d):
        # Game params
        self.p1_states = StochasticGame.generate_states(*p1)
        self.p2_states = StochasticGame.generate_states(*p2)
        self.p1 = p1
        self.p2 = p2
        self.max_depth = d

        # TODO: set real initial state
        s1 = random.choice(list(self.p1_states))
        s2 = random.choice(list(self.p2_states))
        self.root = GameState(s1, s2)

    def initial_state(self):
        return self.root

    def is_terminal(self, s):
        return len(s.history) >= self.max_depth

    def actions(self, s):
        a1 = StochasticGame.get_actions_for(s.s1)
        a2 = StochasticGame.get_actions_for(s.s2)
        return a1, a2

    def step(self, s, a1, a2):
        # TODO: replace with real transition model
        next_s1 = random.choice(list(self.p1_states))
        next_s2 = random.choice(list(self.p2_states))

        # TODO: replace with real reward
        r = random.randint(0, 10)
        reward = (r, -r)

        # Construct next state
        next_history = s.history + [((s.s1, a1), (s.s2, a2))]
        next_s = GameState(next_s1, next_s2, next_history)

        return next_s, reward

if __name__ == '__main__':
    p1, p2 = (1, 2), (1, 2)
    # One day represents one "hand"
    # Depth is based on number of time steps per day
    t = 10 * 60 # 10 mins
    d = (24 * 60 * 60) / t # 10 min steps -> 144 layers
    game = StochasticGame(p1, p2, d)