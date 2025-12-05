from src.fleetctrl.repositioning.RepositioningBase import RepositioningBase
from src.fleetctrl.planning.PlanRequest import PlanRequest

class GameRepositioning(RepositioningBase):
    def __init__(self, fleetctrl, operator_attributes, dir_names):
        super().__init__(fleetctrl, operator_attributes, dir_names)
        # Initialize custom parameters here
        self.utilization_threshold = operator_attributes.get('my_custom_threshold', 0.5)

    def determine_and_create_repositioning_plans(self, sim_time, vid_to_obj, zone_system):
        new_repo_plans = []
        
        # 1. Identify idle vehicles
        idle_vehicles = [v for v in vid_to_obj.values() if not v.assigned_route]
        
        #         
        # 2. Calculate demand/supply balance (Example Logic)
        # This is where your custom algorithm goes (e.g., move to high demand zones)
        for veh in idle_vehicles:
            # Example: Just move everyone to Zone 0 for demonstration
            target_zone_id = 0 
            
            # Create a PlanRequest to move the vehicle
            # You usually need a destination node or position
            destination_node = zone_system.get_random_node(target_zone_id)
            
            # Create the repositioning plan (structure depends on specific FleetPy version)
            # Typically involves creating a PlanRequest with 'repo' status
            repo_plan = PlanRequest(veh.vid, start_loc=veh.pos, end_loc=destination_node, 
                                    time=sim_time, status=PlanRequest.REPO_STATUS)
            
            new_repo_plans.append(repo_plan)

        return new_repo_plans