import random
from game import StochasticGame
from transition_func import load_demand, load_graph, load_zones, Transition

class MonteCarloCFR:
    def __init__(self, game, deep_cfr=False, model=None):
        self.game = game
        self.regret_sum = {} # R[I][a]
        self.strategy_sum = {} # s[I][a]
        self.strategy = {} # S[I][a]

        # Deep CFR handling
        if deep_cfr:
            # Buffers to store data for NN training
            self.regret_samples = [] # RS[I]
            self.policy_samples = [] # PS[I]
            self.model = model
        self.deep_cfr = deep_cfr

    def average_strategy(self):
        for i, strategy in self.strategy_sum.items():
            tot = sum(strategy.values())
            if tot > 0:
                self.strategy[i] = {a: s/tot for a, s in strategy.items()}
            else:
                self.strategy[i] = {a: 1/len(strategy) for a, s in strategy.items()}

    def train(self, iters):
        # Run a traversal for each player for each iteration
        for i in range(iters):
            self.traverse(self.game.initial_state(), p=1, past_u=0.0, past_pi_p1=1.0, past_pi_p2=1.0, past_pi_chance=1.0)
            self.traverse(self.game.initial_state(), p=2, past_u=0.0, past_pi_p1=1.0, past_pi_p2=1.0, past_pi_chance=1.0)

            # Reset game if need be (i.e. end of day/hand)
            self.game.reset_game()
        # Compute the average strategy for each info set
        self.average_strategy()

    def update_infosets(self, s, p):
        # Get infoset, action for player
        i = s.infoset_key(p)
        a = self.game.actions(s)[p-1]
        # Update infosets
        if i not in self.regret_sum:
            self.regret_sum[i] = {_a:0.0 for _a in a}
            self.strategy_sum[i] = {_a:0.0 for _a in a}
        return i, a

    def regret_matching(self, info_set):
        pos_regrets = {a: max(r, 0) for a, r in info_set.items()}
        sum_pos_regrets = sum(pos_regrets.values())
        if sum_pos_regrets > 0:
            sigma = {a: r/sum_pos_regrets for a, r in pos_regrets.items()}
        else:
            sigma = {a: 1/len(pos_regrets) for a, r in pos_regrets.items()}
        return sigma
    
    def counterfactual_regret(self, a, a_sampled, sigma, u,
        pi_total, pi_total_opp_chance, pi_future_player
    ):
        """
        Computes the counterfactual regret for an action a at info set I in trajectory z.

        Args:
        a -- available action at this info set
        a_sampled -- action we actually took at this info set in this trajectory
        sigma -- current strategy for this info set
        u -- actual payoff we obtained at the end of this trajectory
        pi_total -- total traj prob (product of player+opp+chance from root to leaf)
        pi_total_opp_chance -- total traj prob excluding player (product of opp+chance from roof to leaf)
        pi_future_player -- total prob from here to traj end excluding opp+chance (product of player from here to leaf)
        """
        # Calculate importance weighting
        w_I = (u * pi_total_opp_chance * pi_future_player) / pi_total

        # Calculate cf regret
        if a == a_sampled:
            cf_regret = w_I * (1 - sigma[a])
        else:
            cf_regret = -w_I * sigma[a]
        return cf_regret
    
    def update_player(self, infoset, actions, a_sampled, sigma, u,
        pi_total, pi_total_opp_chance, pi_future_player, pi_past_player
    ):
        for a in actions:
            # Get counterfactual regret
            cf_regret = self.counterfactual_regret(a, a_sampled, sigma, u,
                pi_total, pi_total_opp_chance, pi_future_player
            )
            # Update regret
            self.regret_sum[infoset][a] += cf_regret
            # Update strategy
            self.strategy_sum[infoset][a] += (pi_past_player * sigma[a]) / pi_total

            # Handle Deep CFR
            if self.deep_cfr:
                # Add samples for training
                regret_vec = [self.regret_sum[infoset][a] for a in actions]
                self.regret_samples.append((infoset, regret_vec))
                strategy_vec = [self.strategy_sum[infoset][a] for a in actions]
                self.policy_samples.append((infoset, strategy_vec))

    def sample_action(self, actions, sigma):
        return random.choices(actions, weights=[sigma[a] for a in actions], k=1)[0]
    
    def get_regrets(self, s, p, infoset):
        if self.deep_cfr:
            regret_vector = self.model(infoset)
            regrets = {a: r for a, r in zip(self.game.actions(s)[p-1], regret_vector)}
            return regrets
        else:
            return self.regret_sum[infoset]

    def traverse(self, s, p,
        past_u, past_pi_p1, past_pi_p2, past_pi_chance
    ):
        # Base case
        if self.game.is_terminal(s):
            return 0, 1.0, 1.0, 1.0 # future
        
        # Update CFR state for unseen info sets
        infoset_1, actions_1 = self.update_infosets(s, 1)
        infoset_2, actions_2 = self.update_infosets(s, 2)

        # Get strategies for both players
        sigma_1 = self.regret_matching(self.get_regrets(s, 1, infoset_1))
        sigma_2 = self.regret_matching(self.get_regrets(s, 2, infoset_2))

        # Sample actions from strategies
        a1_sampled = self.sample_action(actions_1, sigma_1)
        a2_sampled = self.sample_action(actions_2, sigma_2)

        # Probability of sampled actions
        a1_sampled_prob = sigma_1[a1_sampled]
        a2_sampled_prob = sigma_2[a2_sampled]

        # Sample from chance for next state
        next_s, (r1, r2), chance_prob = self.game.step(s, a1_sampled, a2_sampled)

        # Calculate discounted reward for this node
        curr_u = r1 if p == 1 else r2

        # Recurse to get the future trajectory payoff and prob's
        fut_u, fut_pi_p1, fut_pi_p2, fut_pi_chance = self.traverse(next_s, p,
            past_u + curr_u, past_pi_p1 * a1_sampled_prob, past_pi_p2 * a2_sampled_prob, past_pi_chance * chance_prob
        )

        # Get prob's needed for cf
        total_pi_p1 = past_pi_p1 * a1_sampled_prob * fut_pi_p1
        total_pi_p2 = past_pi_p2 * a2_sampled_prob * fut_pi_p2
        total_pi_chance = past_pi_chance * chance_prob * fut_pi_chance
        pi_total = total_pi_p1 * total_pi_p2 * total_pi_chance
        pi_total_opp_chance = (total_pi_p2 if p == 1 else total_pi_p1) * total_pi_chance

        # Get total utility
        total_u = past_u + curr_u + self.game.gamma * fut_u

        # Update regret for each action in infoset and strategy for infoset
        if p == 1:
            self.update_player(infoset_1, actions_1, a1_sampled, sigma_1, total_u, pi_total, pi_total_opp_chance, fut_pi_p1, past_pi_p1)
        else:
            self.update_player(infoset_2, actions_2, a2_sampled, sigma_2, total_u, pi_total, pi_total_opp_chance, fut_pi_p2, past_pi_p2)

        # Return curr + future payoff and prob's
        return curr_u + fut_u, a1_sampled_prob * fut_pi_p1, a2_sampled_prob * fut_pi_p2, chance_prob * fut_pi_chance

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
    p1, p2 = (1, 5), (1, 5)
    depth = 1

    # Setup transition info
    zones = load_zones('../data/zones/example_zones/example_network/node_zone_info.csv')
    demands = [load_demand('../data/demand/example_demand/matched/example_network/example_100.csv')]
    graph = load_graph('../data/networks/example_network/base/nodes.csv', '../data/networks/example_network/base/edges.csv')
    transition = Transition(p1, p2, graph, zones, demands)

    # Create game
    game = StochasticGame(p1, p2, depth, 600, transition=transition)
    
    # Run MCCFR
    mccfr = MonteCarloCFR(game)
    print("Starting MCCFR Training...")
    mccfr.train(iters=1000)
    print("Training Complete.")

    # Debug
    print_strategy(mccfr)