from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.misc.globals import *

class Player:
    def __init__(self, pid, play, actions):
        self.pid = pid
        self.play = play
        self.actions = actions
        self.total_reward = 0.0

    def action(self):
        return self.actions[-1] # dummy action (everyone moves to last zone)

    def payoff(self, reward, state=None):
        self.total_reward += reward

class Game:
    def __init__(self, players, actions):
        self.actions = actions
        self.players = [Player(pid, play, actions) for pid, play in enumerate(players)]

    def game_step(self, players, state=None):
        # TODO: update internal game state?

        # Give players payoff and get actions for all active players
        actions = []
        for pid, (active, payoff) in enumerate(players):
            player = self.players[pid]
            player.payoff(payoff)
            actions.append(player.action())
        
        return actions

class GameRepositioning(RepositioningBase):
    def __init__(self, fleetctrl, operator_attributes, dir_names):
        # Setup base class
        super().__init__(fleetctrl, operator_attributes, dir_names)
        
        # Setup game
        self.game = Game(self.fleetctrl.sim_vehicles, self.zone_system.get_all_zones())

        # State
        self.vehicle_state = ['idle'] * len(self.fleetctrl.sim_vehicles) # idle, repo, ride

    def is_vehicle_idle(self, vid):
        return not self.fleetctrl.veh_plans[vid].list_plan_stops
    
    def is_vehicle_repositioning(self, vid):
        veh_plan = self.fleetctrl.veh_plans[vid]
        stops = veh_plan.list_plan_stops
        last_stop = stops[-1]
        return last_stop.get_state() == G_PLANSTOP_STATES.REPO_TARGET
    
    def get_vehicle_zone_id(self, veh):
        # Maybe useful for current state?
        zone_id = self.zone_system.get_zone_from_pos(veh.pos)
        return zone_id
    
    def get_pickup_wait_time(self, vid):
        # TODO: is this the correct way to do this?
        for rid, req in self.fleetctrl.rq_dict.items():
            # Check if req is assigned to vid
            if req.get_reservation_flag() or req.service_vehicle != vid:
                continue

            # Get wait time by averaging estimated pickup time
            start_time = req.get_rq_time()
            _, end_time_lo, end_time_up = req.get_o_stop_info()
            wait_time_lo, wait_time_up = end_time_lo - start_time, end_time_up - start_time
            avg_wait_time = (wait_time_up + wait_time_lo) / 2
            return avg_wait_time
        
    def get_vehicle(self, vid):
        return self.fleetctrl.sim_vehicles[vid]

    def reposition_vehicle(self, vid, zone_id):
        list_veh_obj_with_repos = self._od_to_veh_plan_assignment(self.sim_time, None, zone_id, [self.get_vehicle(vid)])
        return [veh_obj.vid for veh_obj in list_veh_obj_with_repos]

    def time_step(self):
        # Get active players and payoffs
        players = []
        for vid, veh in enumerate(self.fleetctrl.sim_vehicles):
            payoff = 0.0 # default when idling or completing a ride
            if self.is_vehicle_idle(vid):
                active = True
                state = 'idle'
            elif self.is_vehicle_repositioning(vid):
                active = False
                state = 'repo'
            else:
                # Completing ride
                active = False
                if self.vehicle_state[vid] != 'ride':
                    # TODO: This may not be the best way to do this?
                    wait_time = self.get_pickup_wait_time(vid)
                    if wait_time is not None:
                        # Assigned a ride so give payoff
                        payoff = -wait_time # neg to incentivize shorter wait time
                        state = 'ride'
                    else:
                        state = 'repo'
            self.vehicle_state[vid] = state
            players.append((active, payoff))

        # Get actions from game
        actions = self.game.game_step(players)

        # Use actions to set repositioning
        list_veh_with_changes = []
        for vid, zone_id in enumerate(actions):
            list_veh_obj_with_repos = self.reposition_vehicle(vid, zone_id)
            list_veh_with_changes.extend(list_veh_obj_with_repos)

        return list_veh_with_changes

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        # Setup
        self.zone_system.time_trigger(sim_time)
        self.sim_time = sim_time
        if lock is None:
            lock = self.lock_repo_assignments

        # Handle repositioning
        list_veh_with_changes = self.time_step()
        
        return list_veh_with_changes