"""
Heuristic functions for A* pathfinding in energy grid optimization.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.models.graph import EnergyGrid
from collections import deque
import numpy as np

class BaseHeuristic(ABC):
    """Base class for heuristic implementations."""
    
    def __init__(self, grid: EnergyGrid):
        """
        Initialize heuristic with grid.
        
        Args:
            grid: The energy distribution grid
        """
        self.grid = grid
    
    @abstractmethod
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost from current to goal node.
        
        Args:
            current: Current station ID
            goal: Goal station ID
            
        Returns:
            float: Estimated cost to goal
        """

    def find_shortest_path_bfs(self, start, end):
        visited = set()
        queue = deque([start])
        visited.add(start)

        parent = {start : None}
        
        while queue:
            current = queue.popleft()

            #If it's the end, it's time to reconstruct the path
            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                
                path.reverse()
                return path
            
            #If current it's not the end, explore all the neighbors
            for neighbor in self.grid.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

                    parent[neighbor] = current


        #If end not found, than there is no path
        return None
    

    def euclidean(self, current: tuple, goal: tuple) -> float:
        """
        Compute euclidean distance between current and goal
        
        Args:
            current: tuple for positions (x,y) of the first node 
            goal: tuple for positions (x,y) of second node

        Returns: 
            float: euclidean distance
        """
        return np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)
    
    def graph_heuristic(self, current: tuple, goal: tuple) -> float:
        #120 = min distance possibile
       return 120 * len(self.find_shortest_path_bfs(current, goal))
    



    def mean_distance_to_goals(self, current_pos: tuple, goals: list) -> float:
        """
        Compute mean euclidean distance between current and a list of other positions

        
        Args:
            current_pos: tuple for positions (x,y) of the first node 
            goals: list (of tuple) for positions (x,y) for the other nodes

        Returns: 
            float: mean euclidean distance
        """
        
        if not goals: 
            return 0.0
        
        distances = []
        for g in goals: 
            d = self.euclidean(current_pos, self.grid.station_data[g]['pos'])
            distances.append(d)
        return np.mean(distances)
    

    def compute_critical_level(self, station: int) -> float:
        """
        Compute the critical level of a station based on its energy level
        
        Args:
            station: station id 

        Returns: 
            float: score

        Example: a station with 80% energy level will get 0.2 as score (low because the station is still full)
        """
        
        energy_level_norm = self.grid.station_data[station]['energy_level']/100
        return 1-energy_level_norm




class EmergencyHeuristic(BaseHeuristic):
    """Heuristic for emergency scenario.
    In this scenario the priority is the time, so the heuristic will be based on "time distance"
    """
    
    def __init__(self, grid: EnergyGrid, critical_station: int):
        """Initialize emergency heuristic."""
        super().__init__(grid)
        self.critical_station = critical_station
        
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering emergency requirements.
        
        TODO: Student Implementation
        Consider:
        - Distance to critical station
        - Energy consumption impact
        - Path criticality
        """

        current_position = self.grid.station_data[current]['pos']
        goal_position = self.grid.station_data[goal]['pos']

        distance = self.graph_heuristic(current, goal)
        # Maximum speed in the system
        max_speed_coeff = 0.5
        # Estimated minimum time to arrive
        h_time = distance * max_speed_coeff

        return h_time
        #return 0

    #def estimate_energy(self, current : int, goal : int) -> float:
        #The consumed energy is 0.2 times the distance between 2 nodes :3

        w = 0.7

        current_position = self.grid.station_data[current]['pos']
        goal_position = self.grid.station_data[goal]['pos']
        distance = self.euclidean(current_position, goal_position)

        return w * distance
        

class MaintenanceHeuristic(BaseHeuristic):
    """Heuristic for maintenance scenario.
    Heuristic will be: euclidean_distance(n,goal)+a*mean_euclidean_distance(n,remaining_goal)

    """
    
    def __init__(self, grid: EnergyGrid, stations_to_visit: List[int]):
        """Initialize maintenance heuristic."""
        super().__init__(grid)
        self.stations_to_visit = stations_to_visit
        
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering maintenance requirements.
        
        TODO: Student Implementation
        Consider:
        - Distance to next station
        - Remaining stations to visit
        - Overall path optimization
        """
 
        return self.graph_heuristic(current, goal)
        
        
class BalancingHeuristic(BaseHeuristic):
    """Heuristic for energy balancing scenario.
    Heuristic will be: euclidean_distance(n,goal)+a*mean_euclidean_distance(n,remaining_goal)
    """
    
    def __init__(self, grid: EnergyGrid, low_energy_stations: List[int]):
        """Initialize balancing heuristic."""
        super().__init__(grid)
        self.low_energy_stations = low_energy_stations
       
        
    def estimate(self, current: int, goal: int) -> float:
        """
        Estimate cost considering balancing requirements.
        
        TODO: Student Implementation
        Consider:
        - Distance to low energy stations
        - Energy levels
        - Optimal connection sequence
        """
        

        return self.graph_heuristic(current, goal)