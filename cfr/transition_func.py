import random
import csv
import math
import networkx as nx
# from game import GameState, StochasticGame # uncomment for testing

def load_graph(node_file, edge_file, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    # Load nodes
    with open(node_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['node_index'])
            G.add_node(
                node_id,
                is_stop_only=row['is_stop_only'].strip().lower() == 'true',
                pos_x=float(row['pos_x']),
                pos_y=float(row['pos_y']),
            )
    # Load edges
    with open(edge_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = int(row['from_node'])
            v = int(row['to_node'])

            # Safely parse source_edge_id
            src = row.get('source_edge_id', '').strip()
            source_edge_id = int(float(src)) if src != '' else None

            G.add_edge(
                u,
                v,
                distance=float(row['distance']),
                travel_time=float(row['travel_time']),
                source_edge_id=source_edge_id,
            )
    return G

def load_zones(file_path):
    zones = {}
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.splitlines()
        for i in range(1, len(lines)):
            line = lines[i].split(',')
            node, zone, _ = int(line[0]), int(line[1]), line[2]
            if zone not in zones:
                zones[zone] = set()
            zones[zone].add(node)
    return zones

def load_demand(file_path):
    demand = []
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.splitlines()
        for i in range(1, len(lines)):
            line = lines[i].split(',')
            t, src, dst, _ = int(line[0]), int(line[1]), int(line[2]), line[3]
            demand.append((t, src, dst))
    return demand

def serve_randomly(remaining, caps):
    '''
    remaining : int (units of demand)
    caps : dict {player_id: capacity}

    Returns:
        updated remaining, updated caps
    '''
    assignments = {1: 0, 2: 0}
    players = [p for p, c in caps.items() if c > 0]

    # Start with log probability of 0 (which is probability 1.0)
    log_prob_assignment = 0.0

    while remaining > 0 and players:
        # The probability of picking this specific player is 1 / len(players)
        # Log prob is -log(len(players))
        log_prob_assignment += -math.log(len(players))

        p = random.choice(players)
        caps[p] -= 1
        assignments[p] += 1
        remaining -= 1

        if caps[p] == 0:
            players.remove(p)

    return remaining, caps, assignments, log_prob_assignment

class Transition:
    def __init__(self, p1, p2, graph, zones, demands):
        '''
        :param zones: dict mapping zone id to set of node ids
        :param demands: list of demand days; demand is a list of t, src, dst
        '''
        self.p1 = p1
        self.p2 = p2
        self.graph = graph
        self.state_history = {}
        self.zones = zones
        self.demands = demands
        self.zone_rep = {
            zid: next(iter(nodes)) for zid, nodes in self.zones.items()
        }

        # Caches
        self.cache = {}
        self.node_to_zone = {node: zid for zid, nodes in self.zones.items() for node in nodes}
        self.zone_dist = {
            (z1, z2): nx.shortest_path_length(self.graph, n1, n2, weight='distance')
            for z1, n1 in self.zone_rep.items() for z2, n2 in self.zone_rep.items()
        }
        
    def travel_time(self, src, dst):
        if (src, dst) in self.cache:
            return self.cache[(src, dst)]
        try:
            dist = nx.shortest_path_length(
                self.graph,
                src,
                dst,
                weight='travel_time',
            )
            if (src, dst) not in self.cache:
                self.cache[(src, dst)] = dist
            return dist
        except nx.NetworkXNoPath:
            return float('inf')
            
    def apply_repositioning(self, player, new_s, a, t):
        """
        Apply repositioning action for a player.

        :param new_s: list of idle vehicles per zone (after demand assignment)
        :param a: repositioning action vector (sum(a) == 0)
        :param t: tuple (time_idx, time_step)
        :return: assignments_log, new_s_post_action
        """
        time_idx, time_step = t
        assignments_log = []

        # Copy state so we can return the updated state
        new_s_post_action = list(new_s)

        out_zones = [(i, -a[i]) for i in range(len(a)) if a[i] < 0]
        in_zones  = [(i,  a[i]) for i in range(len(a)) if a[i] > 0]

        # Sanity check
        if sum(cnt for _, cnt in out_zones) != sum(cnt for _, cnt in in_zones):
            # If action is not flow-conserving, discard it
            return [], new_s_post_action

        # Greedy matching of flows
        oi = ii = 0
        while oi < len(out_zones) and ii < len(in_zones):
            src, out_cnt = out_zones[oi]
            dst, in_cnt  = in_zones[ii]

            flow = min(out_cnt, in_cnt, new_s_post_action[src])  # only take what is available
            if flow == 0:
                # Skip zones with no idle vehicles
                if new_s_post_action[src] == 0:
                    oi += 1
                if in_cnt == 0:
                    ii += 1
                continue

            # Consume idle vehicles at source
            new_s_post_action[src] -= flow

            # Schedule return at destination
            src_zone_id = self.node_to_zone[src]
            dst_zone_id = self.node_to_zone[dst]
            src_node = self.zone_rep[src_zone_id]
            dst_node = self.zone_rep[dst_zone_id]

            pickup_time = self.travel_time(self.zone_rep[oi], src_node)
            trip_time = self.travel_time(src_node, dst_node)
            travel = pickup_time + trip_time
            future_t = time_idx + max(1, int(travel // time_step))

            assignments_log.append(
                (future_t, player, dst + 1, flow)
            )

            out_zones[oi] = (src, out_cnt - flow)
            in_zones[ii]  = (dst, in_cnt - flow)

            if out_zones[oi][1] == 0:
                oi += 1
            if in_zones[ii][1] == 0:
                ii += 1

        return assignments_log, new_s_post_action

    def next_state(self, t, s, a1, a2):
        time_idx, time_step = t

        # Keep track of assignments for vehicles becoming idle again
        assignments_log = []

        # On first time step handle missing vehicles
        if time_idx == 1:
            veh_cnt = {1: sum(s.s1), 2: sum(s.s2)}
            for fut_time_idx in self.state_history:
                for (player, zone_id), count in self.state_history[fut_time_idx].items():
                    veh_cnt[player] += count
            for player in (1,2):
                veh_diff = self.p1[0] - veh_cnt[1] if player == 1 else self.p2[0] - veh_cnt[2]
                for i in range(0, veh_diff):
                    sample_req = random.choice(random.choice(self.demands))
                    req_t, src, dst = sample_req
                    travel = self.travel_time(src, dst)
                    future_t = time_idx + max(1, int(travel // time_step))
                    dst_zone = self.node_to_zone[dst]
                    if dst_zone is None:
                        continue
                    assignments_log.append((future_t, player, dst_zone, 1))
        
        # Sample from demand distribution
        prob_demand_sample = 1.0 / len(self.demands)
        total_log_prob = math.log(prob_demand_sample)
        demand = random.choice(self.demands)

        # Get all the requests between the last time step and this time step
        curr_demand = []
        for req in demand:
            req_t, src, dst = req
            if req_t > time_idx * time_step:
                break
            if time_idx == 1:
                # First time step include everything from beginning to t
                curr_demand.append(req)
            else:
                if time_idx * (time_step-1) <= req_t and req_t <= time_idx * time_step:
                    curr_demand.append(req)
        
        # Aggregate by zone
        demand_by_zones = {zone_id:set() for zone_id in self.zones}
        for req in curr_demand:
            req_t, src, dst = req
            zone_id = self.node_to_zone[src]
            demand_by_zones[zone_id].add(req)

        # Copy state (capacities per zone)
        new_s1, new_s2 = list(s.s1), list(s.s2)

        # Apply completed rides (vehicles becoming idle again)
        if time_idx in self.state_history:
            for (player, zone_id), count in self.state_history[time_idx].items():
                idx = zone_id - 1
                if player == 1:
                    new_s1[idx] += count
                else:
                    new_s2[idx] += count
            del self.state_history[time_idx]

        # Keep track of reward
        reward = {1: 0, 2: 0}

        # For each zone, match demand
        for zone_id, zone_demand in demand_by_zones.items():
            zone_demand = list(zone_demand)
            zone_idx = zone_id - 1
            remaining = len(zone_demand)

            # Match with idle vehicles in same zone first
            caps = {1: new_s1[zone_idx], 2: new_s2[zone_idx]}
            remaining, caps, assigns, log_p_assign = serve_randomly(remaining, caps)
            new_s1[zone_idx] = caps[1]
            new_s2[zone_idx] = caps[2]

            # Add the log prob from this zone to the total
            total_log_prob += log_p_assign

            # Update log
            ride_idx = 0
            for player, cnt in assigns.items():
                for _ in range(cnt):
                    if ride_idx >= len(zone_demand):
                        break
                    req_t, src, dst = zone_demand[ride_idx]
                    ride_idx += 1

                    # Calculate pickup time
                    pickup_time = self.travel_time(self.zone_rep[zone_id], src)
                    # Calculate trip time
                    trip_time = self.travel_time(src, dst)
                    travel = pickup_time + trip_time
                    future_t = time_idx + max(1, int(travel // time_step))
                    dst_zone = self.node_to_zone[dst]

                    if dst_zone is None:
                        continue  # or raise error

                    assignments_log.append((future_t, player, dst_zone, 1))

            # Update reward
            reward[1] += assigns[1]
            reward[2] += assigns[2]

            # If demand met keep going
            if remaining == 0:
                continue

            # Spillover to nearest neighboring zones
            src_node = self.zone_rep[zone_id]

            # Sort other zones by network distance
            other_zones = [(self.zone_dist[(other_id, zone_id)], other_id) for other_id in self.zone_rep if other_id != zone_id]
            other_zones.sort()

            # Serve to other zones
            for _, other_id in other_zones:
                if remaining == 0:
                    break
                idx = other_id - 1
                caps = {1: new_s1[idx], 2: new_s2[idx]}
                remaining, caps, assigns, log_p_assign = serve_randomly(remaining, caps)
                new_s1[idx] = caps[1]
                new_s2[idx] = caps[2]

                # Add the log prob from this spillover event
                total_log_prob += log_p_assign

                # Update log
                for player, cnt in assigns.items():
                    for _ in range(cnt):
                        if ride_idx >= len(zone_demand):
                            break
                        req_t, src, dst = zone_demand[ride_idx]
                        ride_idx += 1

                        pickup_time = self.travel_time(self.zone_rep[other_id], src)
                        trip_time = self.travel_time(src, dst)
                        travel = pickup_time + trip_time
                        future_t = time_idx + max(1, int(travel // time_step))
                        dst_zone = self.node_to_zone[dst]

                        if dst_zone is None:
                            continue  # or raise error

                        assignments_log.append((future_t, player, dst_zone, 1))
                
                # Update reward
                reward[1] += assigns[1]
                reward[2] += assigns[2]

        # Apply repositioning for p1
        assignments_p1, new_s1 = self.apply_repositioning(1, new_s1, a1, t)
        # Apply repositioning for p2
        assignments_p2, new_s2 = self.apply_repositioning(2, new_s2, a2, t)
        # Add all assignments to the log
        assignments_log.extend(assignments_p1)
        assignments_log.extend(assignments_p2)

        # Update history
        for future_t, player, zone_id, cnt in assignments_log:
            if future_t not in self.state_history:
                self.state_history[future_t] = {}
            self.state_history[future_t][(player, zone_id)] = self.state_history[future_t].get((player, zone_id), 0) + 1

        # Return the probability in normal space (exp)
        final_prob = math.exp(total_log_prob)

        return tuple(new_s1), tuple(new_s2), reward[1], reward[2], final_prob

if __name__ == '__main__':
    # Setup transition info
    zones = load_zones('../data/zones/example_zones/example_network/node_zone_info.csv')
    demands = [load_demand('../data/demand/example_demand/matched/example_network/example_100.csv')]
    graph = load_graph('../data/networks/example_network/base/nodes.csv', '../data/networks/example_network/base/edges.csv')
    
    # Uncomment for testing
    # # Setup example state for debugging
    # p, q = 3, len(zones)
    # states = StochasticGame.generate_states(p, q)
    # s = GameState(random.choice(states), random.choice(states))
    # actions = StochasticGame.get_actions_for(s.s1)
    # a1, a2 = random.choice(actions), random.choice(actions)

    # # Test getting next state
    # p1, p2 = (p,q), (p,q)
    # transition = Transition(p1, p2, graph, zones, demands)
    # next_s1, next_s2, r1, r2, prob = transition.next_state((1, 600), s, a1, a2)
    # print(next_s1, next_s2, r1, r2, prob)