from pathlib import Path
import torch
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.misc.globals import *
from cfr.game import GameState, StochasticGame
from cfr.deep_cfr import PolicyNetwork, load_policy, infoset_to_tensor
from cfr.utils import max_actions

# Model info
BASE_DIR = Path(__file__).resolve().parents[3]
NUM_TIME_STEPS = (24 * 60 * 60) / (10 * 60) # 10 mins
NUM_SIM_STEPS = (24 * 60 * 60) / 30 # 30 seconds
# Example network
# MODEL_PATH = BASE_DIR / 'cfr/example_network_policy_5c6z10m.pth'
# MODEL_IN_SIZE = 5 + (NUM_TIME_STEPS * 5 * 2) # 5 zones
# MODEL_OUT_SIZE = max_actions(5, 6) # 5 cars
# NYC network
MODEL_PATH = BASE_DIR / 'cfr/manhattan_network_policy_5c8z10m.pth'
MODEL_IN_SIZE = 8 + (NUM_TIME_STEPS * 8 * 2) # 8 zones
MODEL_OUT_SIZE = max_actions(5, 8) # 5 cars

class GameRepositioning(RepositioningBase):
    def __init__(self, fleetctrl, operator_attributes, dir_names):
        # Setup base class
        super().__init__(fleetctrl, operator_attributes, dir_names)
        
        # Get sim info
        self.player = self.fleetctrl.op_id
        self.vehs = [veh for veh in self.fleetctrl.sim_vehicles if veh.op_id == self.player]
        self.zones = self.zone_system.get_all_zones()
        self.step = 0
        self.time_step_count = 0
        
        # No game state yet
        self.game_state = None
        self.action = None

        # Model
        self.policy_net = PolicyNetwork(in_size=MODEL_IN_SIZE, out_size=MODEL_OUT_SIZE)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_policy(self.policy_net, self.device, MODEL_PATH)

    def get_vehicle_zone_id(self, veh):
        zone_id = self.zone_system.get_zone_from_pos(veh.pos)
        return zone_id
    
    def get_state(self):
        # Get veh dist over zones (state vector)
        state = [0] * (len(self.zones) - 1) # ignore -1 zone
        for veh in self.vehs:
            zone = self.get_vehicle_zone_id(veh)
            state[zone-1] += 1
        
        return state
    
    def update_state(self):
        state = self.get_state()
        
        # Construct new state
        next_history = (self.game_state.history + [((self.game_state.s1, self.action), None)]) if self.game_state is not None else None
        next_s = GameState(state, s2=None, history=next_history) # other player doesn't matter for inference since it's based on infoset

        self.game_state = next_s

    def reposition_vehicle(self, veh, zone_id):
        list_veh_obj_with_repos = self._od_to_veh_plan_assignment(self.sim_time, None, zone_id, [veh])
        return [veh_obj.vid for veh_obj in list_veh_obj_with_repos]

    def action_to_repo_plan(self):
        '''
        Converts a net-flow action plan (counts per zone) into specific vehicle assignments
        minimizing total travel cost.
        '''
        # Get candidate vehicles (veh in zone w/ neg change)
        src_zone_ids = {z_id for z_id, cnt in enumerate(self.action) if cnt < 0}
        candidate_vehs = []
        for vid, veh in enumerate(self.vehs):
            if self.get_vehicle_zone_id(veh) in src_zone_ids:
                candidate_vehs.append((vid, veh))

        # Get dst targets
        # We must 'expand' the destinations. If Zone 5 needs +3 vehicles,
        # we create 3 separate column slots for Zone 5 in the matrix.
        target_zones = []
        for z_id, cnt in enumerate(self.action):
            if cnt > 0:
                target_zones.extend([z_id] * int(cnt))

        # Edge case: If no supply or no demand, return empty plan
        if not candidate_vehs or not target_zones:
            return []

        # Setup routing targets for optimization
        # To save computation, we identify the unique centroids we need to route to.
        unique_dst_zones = set(target_zones)
        # Map zone_id -> (position_tuple). 
        # We use random_centroid_node as the specific target within the zone.
        zone_to_pos = {}
        for z_id in unique_dst_zones:
            node_id = self.zone_system.get_random_centroid_node(z_id)
            zone_to_pos[z_id] = self.routing_engine.return_node_position(node_id)
        
        unique_dst_positions = list(zone_to_pos.values())

        # Construct cost matrix
        # Shape: (num_candidates, num_target_slots)
        # We initialize with a high cost to represent infeasible links if needed.
        cost_matrix = np.zeros((len(candidate_vehs), len(target_zones)))

        for row_idx, (vid, veh) in enumerate(candidate_vehs):
            # Get costs from this vehicle to ALL unique destination centroids at once
            # returns list of tuples: (target_pos, cost, time, dist)
            route_results = self.fleetctrl.routing_engine.return_travel_costs_1toX(
                veh.pos, unique_dst_positions
            )
            
            # Create a lookup for this specific vehicle's costs: {pos: travel_time}
            # Note: Adjust index [2] if your routing engine returns time at a different index
            costs_by_pos = {res[0]: res[2] for res in route_results}

            # Fill the matrix row
            for col_idx, target_z_id in enumerate(target_zones):
                target_pos = zone_to_pos[target_z_id]
                # Use a large default if unreachable, though normally reachable in connected graphs
                cost_matrix[row_idx, col_idx] = costs_by_pos.get(target_pos, 1e9)

        # Solve assignment
        # efficient bipartite matching (Hungarian algorithm)
        # If candidates > targets, it picks the best subset of vehicles.
        # If candidates < targets, it assigns all vehicles to the best available slots.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Formulate plan
        repo_plan = []
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost >= 1e8: 
                continue

            # Get the actual vehicle object and the assigned zone ID
            assigned_veh = candidate_vehs[r][1]
            assigned_zone_id = target_zones[c]
            
            # Structure the plan
            repo_plan.append((assigned_veh, assigned_zone_id))

        return repo_plan

    def time_step(self):
        # Update state
        self.update_state()
        actions = StochasticGame.get_actions_for(self.game_state.s1)

        # Run model to get strategy
        infoset = self.game_state.infoset_key(1)
        infoset_tensor = infoset_to_tensor(infoset, MODEL_IN_SIZE, device=self.device)
        with torch.no_grad():
            output = self.policy_net(infoset_tensor) # 1xN tensor
            logits = output[0] # 1D tensor of size N
            logits = logits[:len(actions)] # mask illegal actions
            strategy = torch.softmax(logits, dim=0) # softmax over 1D vector
        strategy = strategy.cpu().numpy()

        # Sample an action from the strategy
        self.action = random.choices(actions, weights=[strategy[i] for i, a in enumerate(actions)], k=1)[0]

        # Apply action
        repo_plan = self.action_to_repo_plan()
        list_veh_with_changes = []
        for veh, zone_id in repo_plan:
            list_veh_obj_with_repos = self.reposition_vehicle(veh, zone_id)
            list_veh_with_changes.extend(list_veh_obj_with_repos)

        return list_veh_with_changes

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        # Setup
        self.zone_system.time_trigger(sim_time)
        self.sim_time = sim_time
        if lock is None:
            lock = self.lock_repo_assignments

        # Repositioning logic
        list_veh_with_changes = []
        if self.step % (NUM_SIM_STEPS // NUM_TIME_STEPS) == 0 and self.time_step_count < NUM_TIME_STEPS:
            # Only repo for num time steps we trained on
            list_veh_with_changes = self.time_step()
            self.time_step_count += 1
        self.step += 1
        
        return list_veh_with_changes