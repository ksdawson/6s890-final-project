import random
import csv
import networkx as nx
from game import GameState, StochasticGame

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
    assignments = {'p1': 0, 'p2': 0}
    players = [p for p, c in caps.items() if c > 0]

    while remaining > 0 and players:
        p = random.choice(players)
        caps[p] -= 1
        assignments[p] += 1
        remaining -= 1

        if caps[p] == 0:
            players.remove(p)

    return remaining, caps, assignments

class Transition:
    def __init__(self, graph, zones, demands):
        '''
        :param zones: dict mapping zone id to set of node ids
        :param demands: list of demand days; demand is a list of t, src, dst
        '''
        self.graph = graph
        self.state_history = {}
        self.zones = zones
        self.demands = demands
        
    def travel_time(self, src, dst):
        try:
            return nx.shortest_path_length(
                self.graph,
                src,
                dst,
                weight='travel_time',
            )
        except nx.NetworkXNoPath:
            return float('inf')
        
    def get_zone(self, node):
        for zone_id, zone in self.zones.items():
            if node in zone:
                return zone_id

    def next_state(self, t, s, a1, a2):
        time_idx, time_step = t
        
        # Sample from demand distribution
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
            zone_id = self.get_zone(src)
            demand_by_zones[zone_id].add(req)

        # Copy state (capacities per zone)
        new_s1, new_s2 = list(s.s1), list(s.s2)

        # Apply completed rides (vehicles becoming idle again)
        if time_idx in self.state_history:
            for (player, zone_id), count in self.state_history[time_idx].items():
                idx = zone_id - 1
                if player == 'p1':
                    new_s1[idx] += count
                else:
                    new_s2[idx] += count
            del self.state_history[time_idx]

        # Precompute representative node per zone (for distance calc)
        zone_rep = {
            zid: next(iter(nodes)) for zid, nodes in self.zones.items()
        }

        # Keep track of assignments for vehicles becoming idle again
        assignments_log = []

        # For each zone, match demand
        for zone_id, zone_demand in demand_by_zones.items():
            zone_demand = list(zone_demand)
            zone_idx = zone_id - 1
            remaining = len(zone_demand)

            # Match with idle vehicles in same zone first
            caps = {'p1': new_s1[zone_idx], 'p2': new_s2[zone_idx]}
            remaining, caps, assigns = serve_randomly(remaining, caps)
            new_s1[zone_idx] = caps['p1']
            new_s2[zone_idx] = caps['p2']

            # Update log
            ride_idx = 0
            for player, cnt in assigns.items():
                for _ in range(cnt):
                    if ride_idx >= len(zone_demand):
                        break
                    req_t, src, dst = zone_demand[ride_idx]
                    ride_idx += 1

                    travel = self.travel_time(src, dst)
                    future_t = time_idx + max(1, int(travel // time_step))
                    dst_zone = self.get_zone(dst)

                    if dst_zone is None:
                        continue  # or raise error

                    assignments_log.append((future_t, player, dst_zone, 1))

            # If demand met keep going
            if remaining == 0:
                continue

            # Spillover to nearest neighboring zones
            src_node = zone_rep[zone_id]

            # Sort other zones by network distance
            other_zones = []
            for other_id, other_node in zone_rep.items():
                if other_id == zone_id:
                    continue
                try:
                    dist = nx.shortest_path_length(
                        self.graph,
                        src_node,
                        other_node,
                        weight='distance',
                    )
                    other_zones.append((dist, other_id))
                except nx.NetworkXNoPath:
                    continue
            other_zones.sort()

            # Server to other zones
            for _, other_id in other_zones:
                if remaining == 0:
                    break
                idx = other_id - 1
                caps = {'p1': new_s1[idx], 'p2': new_s2[idx]}
                remaining, caps, assigns = serve_randomly(remaining, caps)
                new_s1[idx] = caps['p1']
                new_s2[idx] = caps['p2']

                # Update log
                for player, cnt in assigns.items():
                    for _ in range(cnt):
                        if ride_idx >= len(zone_demand):
                            break
                        req_t, src, dst = zone_demand[ride_idx]
                        ride_idx += 1

                        travel = self.travel_time(src, dst)
                        future_t = time_idx + max(1, int(travel // time_step))
                        dst_zone = self.get_zone(dst)

                        if dst_zone is None:
                            continue  # or raise error

                        assignments_log.append((future_t, player, dst_zone, 1))

        # Apply repositioning actions (simple additive model w/ cap at 0)
        for i in range(len(new_s1)):
            new_s1[i] = max(0, new_s1[i] + a1[i])
            new_s2[i] = max(0, new_s2[i] + a2[i])

        # Update history
        for future_t, player, zone_id, cnt in assignments_log:
            if future_t not in self.state_history:
                self.state_history[future_t] = {}
            self.state_history[future_t][(player, zone_id)] = self.state_history[future_t].get((player, zone_id), 0) + 1

if __name__ == '__main__':
    # Setup transition
    zones = load_zones('../data/zones/example_zones/example_network/node_zone_info.csv')
    demands = [load_demand('../data/demand/example_demand/matched/example_network/example_100.csv')]
    graph = load_graph('../data/networks/example_network/base/nodes.csv', '../data/networks/example_network/base/edges.csv')
    transition = Transition(graph, zones, demands)
    
    # Setup example state for debugging
    p, q = 3, len(zones)
    states = StochasticGame.generate_states(p, q)
    s = GameState(random.choice(states), random.choice(states))
    actions = StochasticGame.get_actions_for(s.s1)
    a1, a2 = random.choice(actions), random.choice(actions)

    # Test getting next state
    next_s = transition.next_state((1, 600), s, a1, a2)