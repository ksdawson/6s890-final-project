
import random
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

def get_combos(target, num_slots, current_combo=None):
    if current_combo is None:
        current_combo = []
    if num_slots == 1:
        if target >= 0:
            yield tuple(current_combo + [target])
        return
    for i in range(target + 1):
        yield from get_combos(target - i, num_slots - 1, current_combo + [i])

def generate_states(p, q):
    return tuple({combo for tot in range(p+1) for combo in get_combos(tot, q)})

def get_actions_for(s):
    p, q = sum(s), len(s)
    new_states = generate_states(p, q)
    actions = {tuple(new_s[i] - s[i] for i in range(q)) for new_s in new_states}
    return tuple(action for action in actions if sum(action) == 0)

class GameState:
    def __init__(self, s1, s2, history=None):
        self.s1 = s1
        self.s2 = s2
        self.history = history or []

class Transition:
    def __init__(self, p1, p2, time_step):
        self.p1_states = generate_states(*p1)
        self.p2_states = generate_states(*p2)
        
        # Store p1 and p2 for later use
        self.p1 = p1
        self.p2 = p2
        self.time_step = time_step

        self.destinations = {}

        num_zones = p1[1]  # q from (p, q)
        self.adjacency_dict = {i: [j for j in range(num_zones) if j != i] 
                               for i in range(num_zones)}
        
        self.demand_df = pd.read_csv('project_data_nyc.csv')
        self.demand_df['counts_per_day'] = self.demand_df['counts_per_day'].apply(eval)

        # self.transition_prob_dist = self.get_transition_prob_dist()
        
    def sample_demand(self, s, time_step):
        """
        Sample demand for a specific time step.
        Aggregates all demands that share the same origin zone and destination zone for that time step.
        
        :param s: object containing demand_df
        :param time_step: the time step to sample from
        :return: tuple of (demand_by_origin, demand_by_dest)
                - both are lists where index is zone and value is sampled count
        """
        # Convert counts from string to list
        demand_df = s.demand_df
        
        # Filter to only rows with matching time step
        time_demands = demand_df[demand_df['time'] == time_step]
        
        # Randomly select one of the 7 days
        day_idx = np.random.randint(0, 7)
        
        # Determine the number of zones (assuming zones are 0-indexed)
        max_zone = max(demand_df['origin_zone'].max(), demand_df['dest_zone'].max())
        num_zones = max_zone + 1
        
        # Initialize lists with zeros
        demand_by_origin = [0] * num_zones
        demand_by_dest = [0] * num_zones
        
        for _, row in time_demands.iterrows():
            origin_zone = row['origin_zone']
            dest_zone = row['dest_zone']
            count = row['counts'][day_idx]
            
            # Aggregate by origin
            demand_by_origin[origin_zone] += count
            
            # Aggregate by destination
            demand_by_dest[dest_zone] += count
        
        return demand_by_origin, demand_by_dest

    def assign_demand_to_operator(self, demand, s_op1, s_op2, zone_index, source_zone_index):
        """
        Assign demand from riders to operators' cars
        """

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
        
        return demand, s_op1, s_op2

    def adjust_action(self, state, action):
        """
        Adjust repositioning due to allocation of demand
        """
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



    def ride_demand_allocation(self, s, s_op1, s_op2, demand):
        """
        Assign demand by zone, first from the same zones of demand, then assigning from adjacent nodes for operators
        """

        # DEMAND ALLOCATION
        # iterate through zones
        for zone_index in range(len(s_op1)):
            # keep assigning cars if there is unmet demand and there are still cars in the zone
            while demand[zone_index] > 0 and (s_op1[zone_index] > 0 or s_op2[zone_index] > 0):
                # if operator has capacity, assign demand to that operator, break ties randomly uniformly
                demand, s_op1, s_op2 = self.assign_demand_to_operator(demand, s_op1, s_op2, zone_index, zone_index)

        # if the while loop is still going, have to assign demand from other squares
        # iterate through zones
        for zone_index in range(len(s_op1)):
            # if unmet demand, and there are still rides anywhere in the network
            while demand[zone_index] > 0 and (s_op1.sum() > 0 or s_op2.sum() > 0):
                
                # find adjacent zones that have available cars from either operator
                available_adj_zones = [
                    z for z in self.adjacency_dict[zone_index] 
                    if s_op1[z] > 0 or s_op2[z] > 0
                ]
                
                if len(available_adj_zones) > 0:
                    # randomly select adjacent zone with available cars to draw from
                    adj_zone_index = np.random.choice(available_adj_zones)
                    demand, s_op1, s_op2 = self.assign_demand_to_operator(demand, s_op1, s_op2, zone_index, adj_zone_index)
                else:
                    # no adjacent zones with available cars, break to avoid infinite loop
                    break

        return demand, s_op1, s_op2


    def reposition(self, s_op1, s_op2, a1, a2):

        # Now apply the adjusted actions
        s_op1 += self.adjust_action(s_op1, a1)
        s_op2 += self.adjust_action(s_op2, a2)

        return s_op1, s_op2

    def select_idle_cars_two_operators(cars_becoming_idle, demand_fulfilled):
        """
        Randomly select cars from cars_becoming_idle for two different operators.
        
        :param cars_becoming_idle: list where index is zone and value is count of idle cars
        :param demand_fulfilled_op1: number of cars needed for operator 1
        :param demand_fulfilled_op2: number of cars needed for operator 2
        :return: tuple of (selected_cars_op1, selected_cars_op2)
                - both are lists where index is zone and value is count of selected cars
        """
        demand_fulfilled_op1, demand_fulfilled_op2 = demand_fulfilled

        # Initialize result lists with zeros
        selected_cars_op1 = [0] * len(cars_becoming_idle)
        selected_cars_op2 = [0] * len(cars_becoming_idle)
        
        # Create a pool of all available car indices
        car_pool = []
        for zone_idx, count in enumerate(cars_becoming_idle):
            car_pool.extend([zone_idx] * count)
        
        # Randomly shuffle the pool
        np.random.shuffle(car_pool)
        
        # Select cars for operator 1 (first demand_fulfilled_op1 cars)
        selected_indices_op1 = car_pool[:demand_fulfilled_op1]
        
        # Select cars for operator 2 (next demand_fulfilled_op2 cars)
        selected_indices_op2 = car_pool[demand_fulfilled_op1:demand_fulfilled_op1 + demand_fulfilled_op2]
        
        # Count how many cars were selected from each zone for operator 1
        for zone_idx in selected_indices_op1:
            selected_cars_op1[zone_idx] += 1
        
        # Count how many cars were selected from each zone for operator 2
        for zone_idx in selected_indices_op2:
            selected_cars_op2[zone_idx] += 1
        
        return selected_cars_op1, selected_cars_op2


    def add_finished_rides(self, s_op1, s_op2, history, t, demand_fulfilled):

        prev_t = t - self.time_step
        car_destinations = self.destinations[prev_t]
        # randomly select the passengers that got picked up from the actual destinations
        cars_becoming_idle1, cars_becoming_idle2 = self.select_idle_cars_two_operators(car_destinations, demand_fulfilled)

        return s_op1 + cars_becoming_idle1, s_op2 + cars_becoming_idle2


    def transition(self, s, a1, a2, t, demand=None):
        # when demand appears in zone, randomly assign cars from operators that have cars in that zone 
        # if there is still demand, randomly assign cars from adjacent zones
        # but how to do that efficiency with an array?

        # assumptions:
        # all zones adjacent to each other
        # all trips take one time step

        # given t, s, a, return the probability of the current state -> next state based on the history

        s_op1, s_op2 = np.array(s.s1), np.array(s.s2)
        a1, a2 = np.array(a1), np.array(a2)

        # sample demand if none passed in, use demand if passed in
        demand, destinations = self.sample_demand(s, t)
        self.destinations[t] = destinations

        # raise exception if sum of action is not 0
        if a1.sum() != 0 or a2.sum() != 0:
            raise ValueError(f"Actions must sum to 0. a1 sum: {a1.sum()}, a2 sum: {a2.sum()}")

        s_op1_inital, s_op2_inital = s_op1.copy(), s_op2.copy()

        # demand and state after demand completely allocated
        demand, s_op1, s_op2 = self.ride_demand_allocation(s, s_op1, s_op2, demand)

        demand_fulfilled = np.sum(s_op1 - s_op1_inital), np.sum(s_op2 - s_op2_inital)

        # calculate reward from chanegs in state
        r = np.sum(s_op1_inital - s_op1).item() - np.sum(s_op2_inital - s_op2).item()
        reward = (r, -r)

        # REPOSITIONING
        s_op1, s_op2 = self.reposition(s_op1, s_op2, a1, a2)

        s_op1, s_op2 = self.add_finished_rides(s_op1, s_op2, s.history, demand_fulfilled)
        # add started rides to demand history
        s.history.append(demand_fulfilled)

        return s_op1, s_op2, reward
    

    def transition_prob_dist(self, t, s, a1, a2, demand):

        s_op1, s_op2 = np.array(s.s1), np.array(s.s2)
        a1, a2 = np.array(a1), np.array(a2)

        # demand and state after demand completely allocated
        demand, s_op1, s_op2 = self.ride_demand_allocation(s, s_op1, s_op2, demand)

        # REPOSITIONING
        s_op1, s_op2 = self.reposition(s_op1, s_op2, a1, a2)

        return tuple(int(x) for x in s_op1), tuple(int(x) for x in s_op2)



    def get_demand_by_origin(self, demand_df, time_step, day_idx, num_zones=4):
        """
        Get total demand by origin zone for a specific time and day.
        
        :param demand_df: DataFrame containing demand data
        :param time_step: the time step to query
        :param day_idx: the day index (0-6) to query
        :param num_zones: number of zones (default 4)
        :return: numpy array where index is zone and value is demand count
        """
        # Filter to only rows with matching time step
        time_demands = demand_df[demand_df['time'] == time_step]
        
        # Initialize array with zeros for all zones
        demand_by_origin = np.zeros(num_zones, dtype=int)
        
        # Aggregate demands by origin zone for the selected day and time step
        for _, row in time_demands.iterrows():
            origin_zone = row['origin_zone']
            count = row['counts'][day_idx]
            demand_by_origin[origin_zone] += count
        
        return demand_by_origin



    def get_transition_prob_dist_single(self, t, s, a1, a2, demand):


        freq = {}
        for d in range(7):
            demand = self.get_demand(self.demand_df, t, d)
            next_s = self.transition_prob_dist(t, s, a1, a2, demand)
            freq[next_s] = freq.get(next_s, 0) + 1

        for next_s in freq:
            freq[next_s] /= sum(freq.values())

        # freq[(s.s1, s.s2, a1, a2, t)] = freq
        return freq




    def get_transition_prob_dist_all(self):

        
        # Calculate total iterations for percentage tracking
        total_s1 = len(self.p1_states)
        total_s2 = len(self.p2_states)
        
        overall_freq = {}
        s1_count = 0
        
        for s1 in self.p1_states:
            s1_count += 1
            s2_count = 0
            
            for s2 in self.p2_states:
                s2_count += 1

                
                for a1 in get_actions_for(s1):
                    for a2 in get_actions_for(s2):
                        for t in self.demand_df['time'].unique():
                            freq = {}
                            for d in range(7):
                                s = GameState(s1, s2)
                                demand = self.get_demand_by_origin(self.demand_df, t, d)
                                next_s = self.transition_prob_dist(t, s, a1, a2, demand)

                                freq[next_s] = freq.get(next_s, 0) + 1

                            for next_s in freq:
                                freq[next_s] /= sum(freq.values())

                            overall_freq[(s1, s2, a1, a2, t)] = freq


        # Save to pickle file
        with open('transition_prob_dist.pkl', 'wb') as f:
            pickle.dump(overall_freq, f)
        
        return freq


    def get_transition_prob_dist_all(self):

        
        # def get_demand(demand_df, time_step, day_idx, num_zones=4):
        #     """
        #     Get total demand by origin zone for a specific time and day.
            
        #     :param demand_df: DataFrame containing demand data
        #     :param time_step: the time step to query
        #     :param day_idx: the day index (0-6) to query
        #     :param num_zones: number of zones (default 4)
        #     :return: numpy array where index is zone and value is demand count
        #     """
        #     # Filter to only rows with matching time step
        #     time_demands = demand_df[demand_df['time'] == time_step]
            
        #     # Initialize array with zeros for all zones
        #     demand_by_origin = np.zeros(num_zones, dtype=int)
            
        #     # Aggregate demands by origin zone for the selected day and time step
        #     for _, row in time_demands.iterrows():
        #         origin_zone = row['origin_zone']
        #         count = row['counts'][day_idx]
        #         demand_by_origin[origin_zone] += count
            
        #     return demand_by_origin


        # Calculate total iterations for percentage tracking
        total_s1 = len(self.p1_states)
        total_s2 = len(self.p2_states)
        
        overall_freq = {}
        s1_count = 0
        
        for s1 in self.p1_states:
            s1_count += 1
            s2_count = 0
            
            for s2 in self.p2_states:
                s2_count += 1

                
                for a1 in get_actions_for(s1):
                    for a2 in get_actions_for(s2):
                        for t in self.demand_df['time'].unique():
                            freq = {}
                            for d in range(7):
                                s = GameState(s1, s2)
                                demand = get_demand(self.demand_df, t, d)
                                next_s = self.transition_prob_dist(t, s, a1, a2, demand)

                                freq[next_s] = freq.get(next_s, 0) + 1

                            for next_s in freq:
                                freq[next_s] /= sum(freq.values())

                            overall_freq[(s1, s2, a1, a2, t)] = freq


        # Save to pickle file
        with open('transition_prob_dist.pkl', 'wb') as f:
            pickle.dump(overall_freq, f)
        
        return freq

        
if __name__ == '__main__':

    # p = number of cars
    # q = number of zones
    p, q = 5, 4
    p1, p2 = (p, q), (p, q) 
    # One day represents one "hand"
    # Depth is based on number of time steps per day
    t = 10 * 60 # 10 mins
    transition = Transition(p1, p2, t)

    df_time = transition.demand_df[transition.demand_df['time'] == 1200]

    # Sum counts by origin zone
    summed = df_time.groupby('dest_zone')['counts_per_day'].apply(
        lambda x: np.sum(np.array(x.tolist()), axis=0)
    )
