import random
from collections import defaultdict
from game import GameState, StochasticGame

class MCCFRTrainer:
    def __init__(self, game):
        self.game = game
        self.regret_sum = defaultdict(float)
        self.strategy_sum = defaultdict(float)
        
        # Maps infoset -> list of valid actions (cached)
        self.infoset_actions = {}

    def get_strategy(self, infoset, legal_actions):
        """Regret Matching: Get current strategy based on accumulated regret."""
        regrets = [self.regret_sum[(infoset, a)] for a in legal_actions]
        positive_regrets = [max(r, 0) for r in regrets]
        sum_positive = sum(positive_regrets)

        if sum_positive > 0:
            return [r / sum_positive for r in positive_regrets]
        else:
            return [1.0 / len(legal_actions) for _ in legal_actions]

    def get_average_strategy(self, infoset, legal_actions):
        """Compute the average strategy over all iterations (the final output)."""
        strategy_sum = [self.strategy_sum[(infoset, a)] for a in legal_actions]
        total = sum(strategy_sum)
        if total > 0:
            return [s / total for s in strategy_sum]
        return [1.0 / len(legal_actions) for _ in legal_actions]

    def train(self, iterations):
        for i in range(iterations):
            # External Sampling: Alternating updates
            # Iteration 1: Update P1 (P2 plays according to strategy)
            # Iteration 2: Update P2 (P1 plays according to strategy)
            self.traverse(self.game.initial_state(), update_player=1)
            self.traverse(self.game.initial_state(), update_player=2)
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{iterations} complete")

    def traverse(self, state, update_player):
        if self.game.is_terminal(state):
            # step() returns (r, -r), so reward depends on who we are
            # We need the reward of the *previous* step that led here.
            # But the reward is returned by step(). 
            # In this architecture, rewards are collected during the traversal returns.
            return 0 

        # 1. Identify infosets and legal actions
        infoset_p1 = state.infoset_key(1)
        infoset_p2 = state.infoset_key(2)
        
        actions_p1, actions_p2 = self.game.actions(state)

        # Cache actions for consistency
        if infoset_p1 not in self.infoset_actions: self.infoset_actions[infoset_p1] = actions_p1
        if infoset_p2 not in self.infoset_actions: self.infoset_actions[infoset_p2] = actions_p2

        # 2. Get strategies for both players
        sigma_1 = self.get_strategy(infoset_p1, actions_p1)
        sigma_2 = self.get_strategy(infoset_p2, actions_p2)

        # 3. Handle External Sampling Logic
        if update_player == 1:
            # P1 is traversing: Iterate all P1 actions, Sample 1 P2 action
            
            # Sample opponent action (P2)
            a2 = random.choices(actions_p2, weights=sigma_2, k=1)[0]
            
            # Values for each of P1's actions
            action_values = []
            node_value = 0

            for i, a1 in enumerate(actions_p1):
                # EXTERNAL SAMPLING:
                # We branch for every one of OUR actions
                next_state, reward_tuple = self.game.step(state, a1, a2)
                
                # Recursion
                # reward_tuple[0] is P1's immediate reward
                future_val = self.traverse(next_state, update_player)
                total_val = reward_tuple[0] + self.game.gamma * future_val
                
                action_values.append(total_val)
                # Weighted contribution to the current node's value
                node_value += sigma_1[i] * total_val

            # Update Regrets for P1
            for i, a1 in enumerate(actions_p1):
                regret = action_values[i] - node_value
                self.regret_sum[(infoset_p1, a1)] += regret
                
            # Update Average Strategy for P2 (The opponent contributes to avg strategy in this pass)
            # NOTE: Standard External Sampling usually updates avg strat for the *traverser*.
            # However, in simultaneous updates, a common pattern is updating the traverser's cumulative strategy.
            for i, a1 in enumerate(actions_p1):
                self.strategy_sum[(infoset_p1, a1)] += sigma_1[i]

            return node_value

        else:
            # P2 is traversing: Iterate all P2 actions, Sample 1 P1 action
            
            # Sample opponent action (P1)
            a1 = random.choices(actions_p1, weights=sigma_1, k=1)[0]
            
            action_values = []
            node_value = 0

            for i, a2 in enumerate(actions_p2):
                next_state, reward_tuple = self.game.step(state, a1, a2)
                
                # Recursion
                # reward_tuple[1] is P2's immediate reward
                future_val = self.traverse(next_state, update_player)
                total_val = reward_tuple[1] + self.game.gamma * future_val
                
                action_values.append(total_val)
                node_value += sigma_2[i] * total_val

            # Update Regrets for P2
            for i, a2 in enumerate(actions_p2):
                regret = action_values[i] - node_value
                self.regret_sum[(infoset_p2, a2)] += regret

            # Update Average Strategy for P2
            for i, a2 in enumerate(actions_p2):
                self.strategy_sum[(infoset_p2, a2)] += sigma_2[i]

            return node_value

def print_strategy(trainer):
    print("\n--- Full Strategy Dump (All Infosets) ---")
    print(f"{'Plyr':<4} | {'State':<10} | {'History Len':<11} | {'Action':<15} | {'Probability'}")
    print("-" * 70)

    # 1. Sort infosets by Player, then by History length (for readability)
    sorted_infosets = sorted(
        trainer.infoset_actions.keys(), 
        key=lambda x: (x[0], len(x[2]), str(x[1]))
    )

    for infoset in sorted_infosets:
        player = infoset[0]
        local_state = infoset[1]
        history = infoset[2]
        
        # Retrieve cached legal actions
        actions = trainer.infoset_actions[infoset]
        
        # Calculate the Nash Equilibrium strategy for this spot
        strategy = trainer.get_average_strategy(infoset, actions)
        
        # Print each action-probability pair
        first_line = True
        for action, prob in zip(actions, strategy):
            # Optional: Filter out 0% probability actions to reduce clutter
            if prob > 0.001:
                state_str = str(local_state)
                hist_len = str(len(history))
                prefix = f"P{player:<3} | {state_str:<10} | {hist_len:<11}" if first_line else " " * 30
                
                print(f"{prefix} | {str(action):<15} | {prob:.4f}")
                first_line = False
                
        if not first_line: # Only print separator if we printed actions
            print("-" * 70)

if __name__ == '__main__':
    # Reduced depth for demonstration speed
    p1, p2 = (1, 2), (1, 2)
    game = StochasticGame(p1, p2, depth=3)
    
    trainer = MCCFRTrainer(game)
    print("Starting MCCFR Training...")
    trainer.train(iterations=1000)
    print("Training Complete.")

    # Debug
    print_strategy(trainer)