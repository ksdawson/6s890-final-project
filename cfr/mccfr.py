import random
from game import StochasticGame

class MonteCarloCFR:
    def __init__(self, game):
        self.game = game
        self.regret_sum = {} # R[I][a]
        self.strategy_sum = {} # s[I][a]
        self.strategy = {} # S[I][a]

    def average_strategy(self):
        for i, strategy in self.strategy_sum.items():
            tot = sum(strategy.values())
            if tot > 0:
                self.strategy[i] = {a:s/tot for a,s in strategy.items()}
            else:
                self.strategy[i] = {a:1/len(strategy) for a,_ in strategy.items()}

    def train(self, iters):
        # Run a traversal for each player for each iteration
        for i in range(iters):
            self.traverse(self.game.initial_state(), p=1, reach_prob=1.0)
            self.traverse(self.game.initial_state(), p=2, reach_prob=1.0)

        # Compute the average strategy for each info set
        self.average_strategy()

    def update_infosets(self, state):
        # Get infoset for each player
        i1 = state.infoset_key(1)
        i2 = state.infoset_key(2)

        # Get actions for each player
        a1, a2 = self.game.actions(state)

        # Update infosets
        if i1 not in self.regret_sum:
            self.regret_sum[i1] = {a:0.0 for a in a1}
            self.strategy_sum[i1] = {a:0.0 for a in a1}
        if i2 not in self.regret_sum:
            self.regret_sum[i2] = {a:0.0 for a in a2}
            self.strategy_sum[i2] = {a:0.0 for a in a2}
        return i1, i2, a1, a2

    def regret_matching(self, info_set):
        pos_regrets = {a:max(r,0) for a,r in info_set.items()}
        sum_pos_regrets = sum(pos_regrets.values())
        if sum_pos_regrets > 0:
            sigma = {a:r/sum_pos_regrets for a,r in pos_regrets.items()}
        else:
            sigma = {a:1/len(pos_regrets) for a,_ in pos_regrets.items()}
        return sigma
    
    def update_player(self, i, a, a_sampled, sigma, u, reach_prob):
        # Counterfactual regret estimator
        for _a in a:
            if _a == a_sampled:
                w = 1.0 / sigma[_a]
            else:
                w = 0.0
            self.regret_sum[i][_a] += w * u - u
        # Average strategy update (importance weighted)
        self.strategy_sum[i][a_sampled] += reach_prob / sigma[a_sampled]

    def traverse(self, s, p, reach_prob):
        # Base case
        if self.game.is_terminal(s):
            # Reward is accumulated at each step in the recursive return value
            return 0
        
        # Update CFR state
        i1, i2, a1, a2 = self.update_infosets(s)

        # Get strategies for both players
        sigma_1 = self.regret_matching(self.regret_sum[i1])
        sigma_2 = self.regret_matching(self.regret_sum[i2])

        # Sample actions from strategies
        a1_sampled = random.choices(a1, weights=[sigma_1[a] for a in a1])[0]
        a2_sampled = random.choices(a2, weights=[sigma_2[a] for a in a2])[0]

        # State transition
        next_s, (r1, r2) = self.game.step(s, a1_sampled, a2_sampled)

        # Recursive continuation
        next_reach = reach_prob * (sigma_2[a2_sampled] if p == 1 else sigma_1[a1_sampled])
        v_next = self.traverse(next_s, p, next_reach)

        # Utility for updating player
        u = (r1 if p == 1 else r2) + self.game.gamma * v_next

        # Update player
        if p == 1:
            self.update_player(i1, a1, a1_sampled, sigma_1, u, reach_prob)
        else:
            self.update_player(i2, a2, a2_sampled, sigma_2, u, reach_prob)

        return u

def print_strategy(trainer):
    # Separate player strategies
    strategy = trainer.strategy
    p1_strategy, p2_strategy = [], []
    for i, s in strategy.items():
        p = i[0]
        p_strategy = p1_strategy if p == 1 else p2_strategy
        p_strategy.append((i, s))
    
    # Sort by history length (i.e. layer)
    p1_strategy.sort(key=lambda x: len(x[0][2]))
    p2_strategy.sort(key=lambda x: len(x[0][2]))

    # Print
    print('Player | Layer | Strategy')
    print('-'*70)
    for i, s in p1_strategy:
        s = {a:round(p,3) for a,p in s.items()}
        print(1, '     |', len(i[2]), '    |', s)
        print('-'*70)
    for i, s in p2_strategy:
        s = {a:round(p,3) for a,p in s.items()}
        print(2, '     |', len(i[2]), '    |', s)
        print('-'*70)

if __name__ == '__main__':
    # Reduced depth for debugging
    p1, p2 = (1, 2), (1, 2)
    game = StochasticGame(p1, p2, depth=3)
    
    # Run MCCFR
    mccfr = MonteCarloCFR(game)
    print("Starting MCCFR Training...")
    mccfr.train(iters=1000)
    print("Training Complete.")

    # Debug
    print_strategy(mccfr)