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
    def __init__(self, s, prev_s=None, next_s=None):
        self.s = s
        self.prev = prev_s
        self.next = next_s

class StochasticGame:
    @classmethod
    def generate_states(cls, p, q):
        return {combo for tot in range(p) for combo in get_combos(tot, q)}

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
        self.d = d

        # TODO: replace with real initial state
        s1 = random.choice(list(self.p1_states))
        s2 = random.choice(list(self.p2_states))

        # State values
        self.p1_curr = GameState(s1)
        self.p2_curr = GameState(s2)
        self.hist_len = 1

    def is_terminal(self):
        return self.hist_len == self.d

    def actions(self):
        a1 = StochasticGame.get_actions_for(self.p1_curr)
        a2 = StochasticGame.get_actions_for(self.p2_curr)
        return a1, a2

    def step(self, state, a1, a2):
        # TODO: replace with real transition model
        p1_next = random.choice(list(self.p1_states))
        p2_next = random.choice(list(self.p2_states))

        # TODO: replace with real reward
        r = random.randint(0, 10)
        reward = (r, -r)

        # Update player states
        self.p1_curr.next = p1_next
        self.p1_curr = GameState(p1_next, prev=self.p1_curr)
        self.p2_curr.next = p2_next
        self.p2_curr = GameState(p2_next, prev=self.p2_curr)
        self.hist_len += 1

        return reward

    def infoset_key(self, player):
        # Player sees only their own local state + their own actions (i.e their state history)
        return self.p1_curr if player == 1 else self.p2_curr

if __name__ == '__main__':
    p1, p2 = (1, 2), (1, 2)
    # One day represents one "hand"
    # Depth is based on number of time steps per day
    t = 10 * 60 # 10 mins
    d = (24 * 60 * 60) / t # 10 min steps -> 144 layers
    game = StochasticGame(p1, p2, d)