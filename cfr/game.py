import random
import numpy as np
import pandas as pd

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
        self.adjacency_dict = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [0, 1, 2]
        }
        self.demand_df = pd.read_csv('project_data_nyc.csv')
        self.demand_df['counts'] = self.demand_df['counts'].apply(eval)

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
        return tuple({tuple(new_s[i] - s[i] for i in range(q)) for new_s in new_states})

    def __init__(self, p1, p2, depth, gamma=1.0):
        # Game params
        self.p1_states = StochasticGame.generate_states(*p1)
        self.p2_states = StochasticGame.generate_states(*p2)
        self.p1 = p1
        self.p2 = p2
        self.max_depth = depth
        self.gamma = gamma

        # Generate random values
        random_vals = np.random.random(4)
        # Scale to sum to n
        arr = (random_vals / random_vals.sum()) * 10 # number of cars for now
        arr = arr.astype(int)  # Convert to int if needed


    def initial_state(self):
        return self.root

    def is_terminal(self, s):
        return len(s.history) >= self.max_depth

    def actions(self, s):
        a1 = StochasticGame.get_actions_for(s.s1)
        a2 = StochasticGame.get_actions_for(s.s2)
        return a1, a2

    def sample_demand(s, time_step):
        """
        Sample demand for a specific time step.
        Aggregates all demands that share the same origin zone for that time step.
        
        :param s: object containing demand_df
        :param time_step: the time step to sample from
        :return: dictionary with origin_zone as key and sampled count as value
        """
        # Convert counts from string to list
        demand_df = s.demand_df
        
        # Filter to only rows with matching time step
        time_demands = demand_df[demand_df['time'] == time_step]
        
        # Randomly select one of the 7 days
        day_idx = np.random.randint(0, 7)
        
        # Aggregate demands by origin zone for the selected day and time step
        demand_by_origin = {}
        for _, row in time_demands.iterrows():
            origin_zone = row['origin_zone']
            count = row['counts'][day_idx]
            
            if origin_zone in demand_by_origin:
                demand_by_origin[origin_zone] += count
            else:
                demand_by_origin[origin_zone] = count
        
        return demand_by_origin


    def step(self, s, a1, a2):

       
        # when demand appears in zone, randomly assign cars from operators that have cars in that zone 
        # if there is still demand, randomly assign cars from adjacent zones
        # but how to do that efficiency with an array?

        # assumptions:
        # 4 zones, all adjacent to each other
        # all trips take one time step
        # 

        time_step = len(s.history)*600

        s_op1, s_op2  = s.s1, s.s2

        # Usage
        demand = self.sample_demand(s, time_step)

        # raise exception if sum of action is not 0
        if a1.sum() != 0 or a2.sum() != 0:
            raise ValueError(f"Actions must sum to 0. a1 sum: {a1.sum()}, a2 sum: {a2.sum()}")


        s_op1_inital = s_op1.copy()
        s_op2_inital = s_op2.copy()

        # DEMAND ALLOCATION
        def assign_demand_to_operator(demand, s_op1, s_op2, zone_index, source_zone_index):
            # if only one operator can handle demand in the same zone, assign to that operator
            if s_op1[source_zone_index] > 0 and s_op2[source_zone_index] == 0:
                demand[zone_index] -= 1
                s_op1[source_zone_index] -= 1
            elif s_op1[source_zone_index] == 0 and s_op2[source_zone_index] > 0:
                demand[zone_index] -= 1
                s_op2[source_zone_index] -= 1
            # if both operators can handle demand in that zone, coin flip to see who gets it
            elif s_op1[source_zone_index] > 0 and s_op2[source_zone_index] > 0:
                operator_demand_assignment = np.random.randint(0, 2)
                if operator_demand_assignment:
                    demand[zone_index] -= 1
                    s_op1[source_zone_index] -= 1
                else:
                    demand[zone_index] -= 1
                    s_op2[source_zone_index] -= 1

        # iterate through zones
        for zone_index in range(len(s_op1)):
            # keep assigning cars if there is unmet demand and there are still cars in the zone
            while demand[zone_index] > 0 and (s_op1[zone_index] > 0 or s_op2[zone_index] > 0):
                # if operator has capacity, assign demand to that operator, break ties randomly uniformly
                assign_demand_to_operator(demand, s_op1, s_op2, zone_index, zone_index)

        # if the while loop is still going, have to assign demand from other squares
        # iterate through zones
        for zone_index in range(len(s_op1)):
            # if unmet demand, and there are still rides anywhere in the network
            # print(s_op1[adj_zone_index].sum(), s_op2[adj_zone_index].sum())
            while demand[zone_index] > 0 and (s_op1.sum() > 0 or s_op2.sum() > 0):
                
                # find adjacent zones that have available cars from either operator
                available_adj_zones = [
                    z for z in s.adjacency_dict[zone_index] 
                    if s_op1[z] > 0 or s_op2[z] > 0
                ]
                
                if len(available_adj_zones) > 0:
                    # randomly select adjacent zone with available cars to draw from
                    adj_zone_index = np.random.choice(available_adj_zones)
                    assign_demand_to_operator(demand, s_op1, s_op2, zone_index, adj_zone_index)
                else:
                    # no adjacent zones with available cars, break to avoid infinite loop
                    break

        print('operator 1', s_op1)
        print('operator 2', s_op2)
        print('repositioning', a1)




        # actual demand that can be satisfied
        r = np.sum(s_op1_inital - s_op1).item() - np.sum(s_op2_inital - s_op2).item()
        reward = (r, -r)

        pre_reposition_s_op1 = s_op1.copy()
        pre_reposition_s_op2 = s_op2.copy()

        # REPOSITIONING

        # action: operator orders a repositioning without knowing how the demand will be 
        # randomly take cars from repositioning to prevent negative states
        def adjust_action(state, action):
            adjusted_action = action.copy()
            
            for i in range(len(state)):
                # check if applying action would result in negative state
                while state[i] + adjusted_action[i] < 0:
                    # Find zones with positive repositioning action
                    available_zones = np.where(adjusted_action > 0)[0]
                    
                    if len(available_zones) > 0:
                        # Randomly pick a zone to steal from
                        steal_from = np.random.choice(available_zones)
                        adjusted_action[steal_from] -= 1
                        adjusted_action[i] += 1
                    else:
                        # No available zones to steal from, break to avoid infinite loop
                        break
            
            return adjusted_action

        # Adjust actions before applying them
        a1 = adjust_action(s_op1, a1)
        a2 = adjust_action(s_op2, a2)

        # Now apply the adjusted actions
        s_op1 += a1
        s_op2 += a2

        print('operator 1 after', s_op1)
        print('operator 2 after', s_op2)


        print('actual repositioning', s_op1 - pre_reposition_s_op1, s_op2 - pre_reposition_s_op2)
        print('final demand', demand)


        # ADD BACK CARS WITH FINISHED RIDES

        # rides started last time will be finsihed rides this time step
        finished_rides1, finished_rides2 = s.history[-1]

        # add started rides to demand history
        demand_fulfilled = (s_op1 - s_op1_inital, s_op2 - s_op2_inital)
        s.history.append(demand_fulfilled)

        s_op1 += finished_rides1
        s_op2 += finished_rides2

        next_s = GameState(s_op1, s_op2, s.history)

        return next_s, reward

if __name__ == '__main__':
    p1, p2 = (1, 2), (1, 2)
    # One day represents one "hand"
    # Depth is based on number of time steps per day
    t = 10 * 60 # 10 mins
    d = (24 * 60 * 60) / t # 10 min steps -> 144 layers
    game = StochasticGame(p1, p2, d)



