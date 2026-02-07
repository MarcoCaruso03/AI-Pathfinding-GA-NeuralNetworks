"""
Maintenance scenario implementation.
Goal: Visit multiple stations in a specific order.
"""

from typing import Optional, List
from src.models.graph import EnergyGrid
from src.algorithm.astar import AStarPathfinder
from src.algorithm.heuristics import MaintenanceHeuristic
from src.utils.report import *


def solve_maintenance_scenario(grid: EnergyGrid, start_station: int,
                             stations_to_visit: List[int],
                             scenario : Dict) -> Optional[List[int]]:
    """
    Find optimal path visiting multiple stations in order.
    
    Args:
        grid: The energy grid
        start_station: Starting station ID
        stations_to_visit: List of stations to visit in order
        
        scenario: Dict contains scenario's details, useful for the txt report
    Returns:
        Optional[List[int]]: Complete maintenance route, if found
    """
    # TODO: Student Implementation
    # 1. Create MaintenanceHeuristic instance
    # 2. Initialize A* pathfinder
    # 3. Consider:
    #    - Find optimal subpaths between consecutive stations
    #    - Ensure stations are visited in the correct order
    #    - Combine subpaths into complete route
    
    heuristic = MaintenanceHeuristic(grid, stations_to_visit)
    
    
    pathfinder = AStarPathfinder(grid, heuristic)
    

    complete_path = []
    current = start_station
    for next_station in stations_to_visit:
        subpath = pathfinder.find_path(current, next_station)
        if not subpath:
            return None
        
        complete_path.extend(subpath[:-1])  # Avoid duplicating intermediate nodes
        current = next_station
    complete_path.append(stations_to_visit[-1])

    # get energy and time info
    time_c = pathfinder.times_cost
    energy_c = pathfinder.energy_cost


    create_txt(pathfinder.explored_nodes, 
               pathfinder.cost,
               pathfinder.length_closed,
               pathfinder.path, 
               time_c, 
               energy_c,
               scenario,
               "ManD.txt", 0)
    
    return complete_path
    