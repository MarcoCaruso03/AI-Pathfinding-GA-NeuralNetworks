"""
Energy balancing scenario implementation.
Goal: Connect low energy stations efficiently.
"""

from typing import Dict, Optional, List
from src.models.graph import EnergyGrid
from src.algorithm.astar import AStarPathfinder
from src.algorithm.heuristics import BalancingHeuristic
from src.models.graph import EnergyGrid
import numpy as np
from src.utils.report import *


def compute_score(grid : EnergyGrid, station: int, goal: int) -> float:
    """
    Compute the score for a possible goal station; used to choose which goal has to be followed in the bilancing scenario
    The score is based on the euclidean distance and critical level. 
    More the score is low => more urgent is that station

    Args:
        station: station id, current station 
        goal: possible station goal id

    Returns: 
        float: score

    """
    current_position = grid.station_data[station]['pos']
    goal_position = grid.station_data[goal]['pos']
    return np.sqrt((current_position[0] - goal_position[0])**2 + (current_position[1] - goal_position[1])**2)/(1-grid.station_data[goal]['energy_level']/100)

def set_goal(grid: EnergyGrid, station: int, possible_goal: List[int]) -> int: 
    """
    Return the goal with the lower score. It's used to choose which is the goal for each A*

    Args:
        grid: 
        station: current node 
        possible_goals: list of possible future goals 

    Returns: 
        int: choosen goal's id
    """
    
    scores = []
    #print("possible goal (da set_goal): ", possible_goal)
    for g in possible_goal: 
        scores.append(compute_score(grid, station, g))
    score_min = np.argmin(scores)
    return possible_goal[score_min]





def solve_balancing_scenario(grid: EnergyGrid, start_station: int,
                           low_energy_stations: List[int],
                           scenario : Dict) -> Optional[List[int]]:
    """
    Find optimal path connecting low energy stations.
    
    Args:
        grid: The energy grid
        start_station: Starting station ID
        low_energy_stations: List of stations needing energy
        
        scenario: Dict contains scenario's details, useful for the txt report
    Returns:
        Optional[List[int]]: Path connecting all required stations, if found
    """
    # TODO: Student Implementation
        # 1. Create BalancingHeuristic instance
    #heuristic = BalancingHeuristic(grid, low_energy_stations)
    
    # 2. Initialize A* pathfinder
    #pathfinder = AStarPathfinder(grid, heuristic)
    
    # 3. Consider:
    #    - Find optimal order to visit stations
    #    - Minimize total path distance
    #    - Consider energy levels when planning route
    #    - Implement nearest neighbor or similar approach
    # 4. Example approach:
    # ordered_stations = find_optimal_station_order(grid, start_station, low_energy_stations)
    # return find_path_through_stations(pathfinder, ordered_stations)
     
    complete_path = []
    current = start_station
    remaining_goals = low_energy_stations.copy()
    
    # for the report
    total_cost = 0
    all_explored = []
    total_time = 0
    total_energy = 0

    while remaining_goals:
        
        # select the current A* goal
        g_star = set_goal(grid, current, remaining_goals)
        
        heuristic = BalancingHeuristic(grid, remaining_goals)

        pathfinder = AStarPathfinder(grid, heuristic)
        
        
        subpath = pathfinder.find_path(current, g_star)

        total_cost += pathfinder.cost
        all_explored.extend(pathfinder.explored_nodes)

        
        
        if not subpath:
            return None

        complete_path.extend(subpath[:-1])  # Avoid duplicating intermediate nodes
        
        # update the current node    
        current = g_star
        # remove the goal from the list
        remaining_goals.remove(g_star)
    complete_path.append(current)

    #complete_path.append(low_energy_stations[-1])

    # get time and energy costs, for the report
    time_c = pathfinder.times_cost
    energy_c = pathfinder.energy_cost

    
    # create the report
    create_txt(
        all_explored,
        total_cost,
        len(all_explored),
        complete_path,
        time_c,
        energy_c,
        scenario,
        "BalD.txt",
        0
    )


    return complete_path

