"""
Emergency scenario implementation.
Goal: Reach the most critical station in minimum time.
"""

from typing import Dict, Optional, List
from src.models.graph import EnergyGrid
from src.algorithm.astar import AStarPathfinder
from src.algorithm.heuristics import EmergencyHeuristic
from src.utils.report import *

def solve_emergency_scenario(grid: EnergyGrid, start_station: int, 
                           critical_station: int,
                           scenario : Dict) -> Optional[List[int]]:
    """
    Find the optimal path from start to critical station.
    
    Args:
        grid: The energy grid
        start_station: Starting station ID
        critical_station: Critical station ID to reach
        
        scenario: Dict contains scenario's details, useful for the txt report
    Returns:
        Optional[List[int]]: Path from start to critical station, if found
    """
    # TODO: Student Implementation
    # 1. Create EmergencyHeuristic instance
    # 2. Initialize A* pathfinder with the heuristic
    # 3. Consider:
    #    - Minimize time to reach critical station
    #    - Account for energy consumption
    #    - Handle path constraints
    # 4. Find and return the optimal path
        # get energy and time info
    heuristic = EmergencyHeuristic(grid, critical_station)
    
    
    pathfinder = AStarPathfinder(grid, heuristic, 1)


    path = pathfinder.find_path(start_station, critical_station)
    
    # get time and energy costs for the report
    time_c = pathfinder.times_cost
    energy_c = pathfinder.energy_cost


    create_txt(pathfinder.explored_nodes, 
               pathfinder.cost,
               pathfinder.length_closed,
               pathfinder.path, 
               time_c, 
               energy_c,
               scenario,
               "EmP1.txt", 2, pathfinder.stringdeb)

    print("DEBUG:\n",pathfinder.stringdeb)

    return path
    