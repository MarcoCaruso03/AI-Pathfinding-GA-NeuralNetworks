"""
A* pathfinding algorithm implementation for energy grid optimization.
"""

import heapq
from typing import List, Optional, Dict, Set, Callable
from queue import PriorityQueue
from src.models.graph import EnergyGrid
from src.algorithm.heuristics import BaseHeuristic
from src.utils.report import *

class AStarPathfinder:
    """A* pathfinding implementation for energy grid."""
    
    def __init__(self, grid: EnergyGrid, heuristic: BaseHeuristic, flag = 0):
        """
        Initialize pathfinder with grid and heuristic.
        
        Args:
            grid: The energy distribution grid
            heuristic: Heuristic function implementation
            flag: 1 if emergency case
        """
        self.grid = grid
        self.heuristic = heuristic
        self.case = flag

        # POSSIBLE TIME OR ENERGY CONSTRAINTS
        self.T_MAX = 400
        self.E_MAX = 800

        # for A* result in the txt report
        self.cost = 0
        self.length_closed = 0
        self.length_path = 0
        self.explored_nodes = []
        self.path = []
        self.total_e = 0
        self.total_t = 0

        # generate time costs and energy costs (used for emergency case)
        self.times_cost, self.energy_cost = self.get_costs()
    
        # debug string for emergency case (will contain all the discarded paths)
        self.stringdeb = ""


    def get_costs(self):
        """
        Create dictionaries for cost and time, based on the type of edge,
        used in emergencies, for constraints

        The assumption is: a long road (wight=180) will be a longer way but faster and more expensive in terms of energy (like highways)
        instead a short way (wight=120) it will take longer but less energy 
        """
        # Coefficients which models different type of roads
        coef_stard_e_time = 1
        coef_stard_e_energy = 0.8
        coef_crossa_e_time = 0.5
        coef_crossa_e_energy = 1.4


        time_cost = {}
        energy_cost = {}
        for u, v, w in self.grid.graph.edges(data='weight'):
            if w == 120: #classic edge
                time_cost[(u,v)] = coef_stard_e_time * w
                time_cost[(v,u)] = coef_stard_e_time * w
                energy_cost[(u,v)] = coef_stard_e_energy * w
                energy_cost[(v,u)] = coef_stard_e_energy * w
            else:
                time_cost[(u,v)] = coef_crossa_e_time * w
                time_cost[(v,u)] = coef_crossa_e_time * w
                energy_cost[(u,v)] = coef_crossa_e_energy * w
                energy_cost[(v,u)] = coef_crossa_e_energy * w
            
        return time_cost, energy_cost


    def find_path(self, start: int, goal: int) -> Optional[List[int]]:
        """
        Find optimal path between start and goal stations.
        
        Args:
            start: Starting station ID
            goal: Goal station ID
            
        Returns:
            Optional[List[int]]: Path from start to goal if found, None otherwise
        """
        # TODO: Student Implementation

        # maintenance, energy balancing scenario => "classic A*" => using just the node as state and distance to min
        if self.case == 0:

            # 2. Initialize algorithm
            #    - Add start node to open set
            #    - Set initial g_score
            #    - Set initial f_score using heuristic
            open = []
            heapq.heappush(open, (0, start))

            g_score = {start: 0}
            f_score = {start: self.heuristic.estimate(start, goal)}

            # closed, parent
            parent = {start: None}
            closed = set()
            

            # 3. Main loop
            #    - Get node with lowest f_score from open set
            #    - If goal reached, reconstruct path
            #    - For each neighbor:
            #      * Calculate tentative g_score
            #      * If better path found, update data structures
            while open: 
                # Get the top node (lowest f_score)
                current_f, current = heapq.heappop(open)
                if current in closed: 
                    continue
                closed.add(current)
                if current == goal:
                    # 4. Reconstruct path when goal is reached
                    path = self._reconstruct_path(parent, current)
                    
                    # update internal variables used for the report generation

                    self.explored_nodes += closed
                    #print("NODI ESPLORATI:",self.explored_nodes)

                    self.length_closed += len(closed)
                    #print("# NODI ESPLORATI:",self.length_closed)

                    self.cost += compute_total_cost(self.grid, path)
                    #print("COSTO:",self.cost)


                    self.path += path
                    self.length_path += len(path)
                    #print("LUNGHEZZA PERCORSO:",self.cost)

                    return path
                
            
                for neighbor in self.grid.graph.neighbors(current):
                    if neighbor in closed: 
                        continue
                    # the cost is the weight for the edge so the distance (for these 2 scenarios)
                    tentative_g = g_score[current] +  self.grid.graph[current][neighbor]['weight']

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        parent[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g+self.heuristic.estimate(neighbor, goal)

                        heapq.heappush(open, (f_score[neighbor], neighbor))
            # 5. Return None if no path found
            return None
    

        else: 
            # emergency case

            start_state = (start, 0)  # (ID, Energy)

            
            start_h = self.heuristic.estimate(start, goal)
            open_set = []
            heapq.heappush(open_set, (start_h, 0, start_state))
            
    
            # parent Dict: in order to get all the path
            came_from = {start_state: None}
            g_score = {start_state: 0}

            # Dict for pruning 
            # Maps NodeID -> Minimum Energy We've Ever Got Here With
            min_energy_at_node = {}

            print("=== AVVIO A* EMERGENZA===")
            while open_set:
                # Extraction: A* guarantees that we extract the path with the smallest F (most promising in time)
                current_f, current_time, current_state = heapq.heappop(open_set)
                current_node, current_energy = current_state


                self.explored_nodes.append(current_node)

                # Check if it's the goal
                if current_node == goal:
                    # update internal variables used for the report generation

                    path = self._reconstruct_path_from_states(came_from, current_state)
                    self.length_closed = len(set(self.explored_nodes))
                    #print("# NODI ESPLORATI:",self.length_closed)

                    self.cost += compute_total_cost(self.grid, path)
                    #print("COSTO:",self.cost)


                    self.path += path
                    self.length_path += len(path)
                    #print("LUNGHEZZA PERCORSO:",self.cost)

                    return path
                
                # Dominance Check (To avoid infinite loops and long times)
                # If we've already reached this node with LESS (or the same) energy.
                # Since A* extracts in time order (or estimate), this path is worse in both time and energy => skip
                if current_node in min_energy_at_node:
                    if min_energy_at_node[current_node] <= current_energy:
                        continue
                # Update the best energy seen for this station
                min_energy_at_node[current_node] = current_energy

                # neighbor
                for neighbor in self.grid.graph.neighbors(current_node):
                    # compute edge's cost
                    edge_time = self.times_cost.get((current_node, neighbor))
                    edge_energy = self.energy_cost.get((current_node, neighbor))
            
                    tentative_time = current_time + edge_time
                    tentative_energy = current_energy + edge_energy

                    # Check constrains
                    if tentative_time > self.T_MAX or tentative_energy> self.E_MAX:
                        path_partial = self._reconstruct_path_from_states(came_from, current_state) + [neighbor]
                        self.stringdeb += f"Percorso scartato: {path_partial} (tempo={tentative_time}, energia={tentative_energy})\n"
                        continue


                    # Create a new state (NeighID, Energy)
                    neighbor_state = (neighbor, tentative_energy)
                    
                    # Updating and Queuing
                    # First time we've seen this specific status or improved the time
                    if neighbor_state not in g_score or tentative_time < g_score[neighbor_state]:
                        came_from[neighbor_state] = current_state
                        g_score[neighbor_state] = tentative_time

                        h = self.heuristic.estimate(neighbor, goal)
                        f = tentative_time + h

                        heapq.heappush(open_set, (f, tentative_time, neighbor_state))
            return None

        
    def _reconstruct_path(self, came_from: Dict[int, int], 
                         current: int) -> List[int]:
        """
        Reconstruct path from came_from dictionary.
        
        Args:
            came_from: Dictionary tracking path predecessors
            current: Current (goal) node
            
        Returns:
            List[int]: Reconstructed path
        """
        # TODO: Student Implementation
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    
    def _reconstruct_path_from_states(self, came_from, current_state):
        """
        Reconstructs the path from a dictionary based on extended states.
        """
        path = []
        
        while current_state is not None:
            # Take just the id
            node_id = current_state[0]
            path.append(node_id)
            
            # Go back to the parent
            current_state = came_from[current_state]
            
        # reverse
        path.reverse()
        return path