
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

def generate_states(p, q):
    return tuple({combo for tot in range(p+1) for combo in get_combos(tot, q)})

@classmethod
def get_actions_for(s):
    p, q = sum(s), len(s)
    new_states = generate_states(p, q)
    actions = {tuple(new_s[i] - s[i] for i in range(q)) for new_s in new_states}
    return tuple(action for action in actions if sum(action) == 0)

class Transition:
    def __init__(self, p1, p2, t):
        self.p1_states = generate_states(*p1)
        self.p2_states = generate_states(*p2)
        
        # Store p1 and p2 for later use
        self.p1 = p1
        self.p2 = p2
        self.time_step = t
        
    def sample_demand(self, s, time_step):
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


    def add_finished_rides(self, s_op1, s_op2, history):


        s_op1 += history[-1][0]
        s_op2 += history[-1][1]

        return s_op1, s_op2


    def transition(self, s, a1, a2, demand=None):
        # when demand appears in zone, randomly assign cars from operators that have cars in that zone 
        # if there is still demand, randomly assign cars from adjacent zones
        # but how to do that efficiency with an array?

        # assumptions:
        # 4 zones, all adjacent to each other
        # all trips take one time step


        # given t, s, a, return the probability of the current state -> next state based on the history

        s_op1, s_op2 = np.array(s.s1), np.array(s.s2)
        a1, a2 = np.array(a1), np.array(a2)

        # sample demand if none passed in, use demand if passed in
        if demand == None:
            demand = self.sample_demand(s, self.time_step)


        # pre compute state transition probabilies by claculating over all times in your demand


        # for day in all days:
            # demand = demand[t, day] s, a1, a2


        # raise exception if sum of action is not 0
        if a1.sum() != 0 or a2.sum() != 0:
            raise ValueError(f"Actions must sum to 0. a1 sum: {a1.sum()}, a2 sum: {a2.sum()}")

        s_op1_inital, s_op2_inital = s_op1.copy(), s_op2.copy()

        # demand and state after demand completely allocated
        demand, s_op1, s_op2 = self.ride_demand_allocation(s, s_op1, s_op2, demand)

        # calculate reward from chanegs in state
        r = np.sum(s_op1_inital - s_op1).item() - np.sum(s_op2_inital - s_op2).item()
        reward = (r, -r)

        # REPOSITIONING
        s_op1, s_op2 = self.reposition(s_op1, s_op2, a1, a2)


        if len(s.history) > 0: 
            s_op1, s_op2 = self.add_finished_rides(s_op1, s_op2, s.history)


        # i need to apply where the destination zone fo each demand is

        # add started rides to demand history
        demand_fulfilled = (s_op1 - s_op1_inital, s_op2 - s_op2_inital)
        s.history.append(demand_fulfilled)






        return s_op1, s_op2, reward
    


    def transition_prob_dist(self, s):



        def get_demand(self, demand_df, time_step, day_idx):
            """
            Get total demand by origin zone for a specific time and day.
            
            :param demand_df: DataFrame containing demand data
            :param time_step: the time step to query
            :param day_idx: the day index (0-6) to query
            :return: dictionary with origin_zone as key and total demand count as value
            """
            # Filter to only rows with matching time step
            time_demands = demand_df[demand_df['time'] == time_step]
            
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


        # for all possible combindtions of state, action, time

        demand = s.demand_df

        freq = {}
        for s1 in self.p1_states:
            for s2 in self.p2_states:
                for a1 in get_actions_for(s1):
                    for a2 in get_actions_for(s2):
                        
                        for t in demand['time'].unique():
                            for d in range(7):
                                demand = get_demand(demand, t, d)
                                next_s = self.transition(t, s, a1, a2, demand=demand[d])

                                freq[(t, s, a1, a2, demand)] = next_s

                                freq[next_s] += 1
                                for next_s in freq:
                                    freq[next_s] \= total