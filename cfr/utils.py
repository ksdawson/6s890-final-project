import math
import time
from cfr.game import StochasticGame

def time_func(func, args):
    start = time.time()
    func(**args)
    end = time.time()
    print(f'Ran in {round(end-start)} s')

def state_space_size(p, q):
    # Number of ways to assign 0 ... p items to q buckets
    return math.comb(p + q, q)

def max_actions(p, q):
    return math.comb(p + q - 1, q - 1)
    
def get_infoset_count(p, q, depth):
    # Calculate State Space K
    K = state_space_size(p, q)
    
    # Calculate Branching Factor B (Actions * Observations)
    # Note: This assumes Max Actions = K. 
    # In your specific code, actions are limited by sum(s), so this is an upper bound.
    B = K * K
    
    # Geometric Series Sum: a * (r^n - 1) / (r - 1)
    # a = K (initial width)
    # r = B (growth factor)
    # n = depth
    if B == 1:
        return K * depth
        
    total_infosets = K * (B**depth - 1) // (B - 1)
    return total_infosets

if __name__ == '__main__':   
    p, q = 10, 5
    d = 24
    count = get_infoset_count(p, q, d)
    # states = StochasticGame.generate_states(p, q)
    # num_actions = 0
    # for state in states:
    #     actions = StochasticGame.get_actions_for(state)
    #     num_actions = max(num_actions, len(actions))
    # num_states = len(states)

    print(f"Game Params: P={p}, Q={q}, Depth={d}")
    print(f"State Space Size (K): {state_space_size(p, q)}")
    # print(f'Num states, actions: {num_states}, {num_actions}')
    print(f"Total Infosets: {count:,}")