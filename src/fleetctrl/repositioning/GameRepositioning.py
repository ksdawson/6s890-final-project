from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.fleetctrl.planning.PlanRequest import PlanRequest

class Player:
    def __init__(self, vid, veh):
        self.vid = vid
        self.veh = veh

    def action(self):
        pass

    def payoff(self):
        pass

class Game:
    def __init__(self, vehicles=[]):
        # Setup players
        self.players = []
        for vid, veh in enumerate(vehicles):
            player = Player(vid, veh)
            self.players.append(player)

class GameRepositioning(RepositioningBase):
    def __init__(self, fleetctrl, operator_attributes, dir_names):
        # Setup base class
        super().__init__(fleetctrl, operator_attributes, dir_names)
        
        # Setup game
        self.game = Game(self.fleetctrl.sim_vehicles)

    def is_vehicle_idle(self, vid):
        return not self.fleetctrl.veh_plans[vid].list_plan_stops

    def determine_and_create_repositioning_plans(self, sim_time, lock=None):
        # Setup
        self.zone_system.time_trigger(sim_time)
        self.sim_time = sim_time
        if lock is None:
            lock = self.lock_repo_assignments
        
        return []