from typing import Dict, List
from src.models.graph import EnergyGrid

"""
This .py contains useful function used for the generation of the txt report 
"""


def compute_total_cost(grid: EnergyGrid, path: List[int]) -> float:
    """Calculate the total cost of a path.
    
    Args: 
        grid
        path
    
    Return: 
        float: cost of the path. As sum of the edge's weight
    """
    total_cost = 0.0
    if not path:
        return total_cost
        
    for i in range(len(path) - 1):
        edge_data = grid.graph.get_edge_data(path[i], path[i + 1])
        if edge_data and 'weight' in edge_data:
            total_cost += edge_data['weight']
            
    return total_cost

def modify_path(path: List[int]) -> List[int]:
    """
    The function will modify the path removing two consecutive equal nodes, given by the merge of two subpaths (from different A*)

    Args:
        path: the original path 

    Return: 
        List[int] : modified path
    """
    if not path:
        return path

    new_path = [path[0]]
    for node in path[1:]:
        if node != new_path[-1]:
            new_path.append(node)
    return new_path

def compute_total_time_cost(path: List[int], times_c : Dict, energy_c : Dict): 
    """
    Compute the total TIME and ENERGY cost given a path and costs dict

    Args: 
        path
        times_c : Dict contains the time cost for each pair of nodes 
        energy_c : Dict contains the energy cost for each pair of nodes 

    Returns: 
        float, float: costs 
    """
    time_tot = 0
    energy_tot = 0
    for i in range(len(path)-1):
        u = path[i]
        v = path[i+1]
        time_tot+=times_c[(u,v)]
        energy_tot+=energy_c[(u,v)]
    return time_tot, energy_tot

def create_txt(
        explored_node : List,
        cost : int, 
        length_c : int, 
        p : List[int],
        time_c : Dict, 
        energy_c : Dict,
        scenario : Dict,
        file_path,
        flag_scenario: int,
        deb_strin = ""):
    
    """
    Generate the txt report of a simulation. 

    Args: 
        explored_node : List of the node explored during the simulation
        cost : cost of the path (sum of the edge's weight)
        length_c : # of noded explored
        p : path
        time_c : dict with time costs
        energy_c : dict with energy costs
        scenario : dict with scenario's info
        file_path :
        flag_scenario : 0 for maintenance and balancing scenario, 2 for emergency
        deb_string : debug string
    """

    with open(file_path, "w") as f:

        # get useful information from the scenario dict
        title = scenario["description"]
        goals = scenario["targets"]
        start = scenario["start"]
        cons = scenario["constraints"]

        # Remove identical double nodes between the end and start of two consecutive subpaths 
        p_mod=modify_path(p)
        time_tot, energy_tot = compute_total_time_cost(p_mod, time_c, energy_c)
        f.write(f"REPORT A*: scenario {title}\n")
        f.write(f"Start:{start}\n")
        f.write(f"Goals:{goals}\n")
        f.write(f"Constraints:{cons}\n")
        f.write(f"====================\n")

        if flag_scenario == 2 and p_mod==[]: #emergency and unmet constraints
            f.write("VALID PATH NOT FOUND\n")
            f.write("Below are the routes found and which constraints they would have violated:\n\n")
            f.write(deb_strin)
        else:
            # man/balance scenario && viconcolo on the distance not satisfied
            if 'max_total_distance' in cons and cost > cons['max_total_distance']:
                f.write("VALID PATH NOT FOUND\n")
                f.write("IMPOSSIBLE TO MEET THE DISTANCE CONSTRAINT\n")
            else: #man/bil scenario with distance constraint ok or emergency ok constraints
                f.write(f"# Explored nodes: {length_c}\n")
                f.write(f"Explored nodes: {explored_node}\n")
                f.write(f"Path: {p_mod}\n")        
                f.write(f"Solution cost: {cost}\n")
                f.write(f"Path length: {len(p_mod)}\n")
                f.write(f"Time spent: {time_tot}\n")
                f.write(f"Energy spent: {energy_tot}\n")

                if deb_strin != "":
                    f.write(f"====================\n")
                    f.write("Below are the routes found and which constraints they would have violated:\n\n")
                    f.write(deb_strin)
        
    