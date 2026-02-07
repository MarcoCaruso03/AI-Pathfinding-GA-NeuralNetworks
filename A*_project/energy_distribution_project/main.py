import argparse
import logging
from typing import List, Dict, Tuple, Optional
import sys

sys.dont_write_bytecode = True

from src.models.graph import EnergyGrid
from src.utils.visualization import GridVisualizer
from src.scenarios.emergency import solve_emergency_scenario
from src.scenarios.maintenance import solve_maintenance_scenario
from src.scenarios.balancing import solve_balancing_scenario

def setup_logging() -> None:
    """Configure logging for the application."""
    class CustomFormatter(logging.Formatter):
        """Custom formatter with colors and symbols"""
        
        grey = "\x1b[38;20m"
        blue = "\x1b[34;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        def __init__(self):
            super().__init__()
            self.FORMATS = {
                logging.DEBUG: self.grey + "🔍 DEBUG: %(message)s" + self.reset,
                logging.INFO: self.blue + "ℹ️  %(message)s" + self.reset,
                logging.WARNING: self.yellow + "⚠️  WARNING: %(message)s" + self.reset,
                logging.ERROR: self.red + "❌ ERROR: %(message)s" + self.reset,
                logging.CRITICAL: self.bold_red + "🚨 CRITICAL: %(message)s" + self.reset
            }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger.handlers = []
    logger.addHandler(ch)

def calculate_path_cost(grid: EnergyGrid, path: List[int]) -> float:
    """Calculate the total cost of a path."""
    total_cost = 0.0
    if not path:
        return total_cost
        
    for i in range(len(path) - 1):
        edge_data = grid.graph.get_edge_data(path[i], path[i + 1])
        if edge_data and 'weight' in edge_data:
            total_cost += edge_data['weight']
            
    return total_cost

def get_scenario_data(scenario_type: str) -> dict:
    """Get predefined data for each scenario."""
    scenarios = {
        'emergency': {
            'start': 1,  # Adjacent to Central station
            'targets': [17],  # Most critical station (West)
            'constraints': {
                'max_energy_consumption': 800,
                'max_time': 400
            },
            'description': 'Emergency response to critical station'
        },
        'maintenance': {
            'start': 6,  # North-East station
            'targets': [5, 12, 8],  # Stations to visit in order
            'constraints': {
                'visit_order': True,
                'max_total_distance': 1800
            },
            'description': 'Planned maintenance route for 3 stations'
        },
        'balancing': {
            'start': 11,  # South extreme station
            'targets': [5, 10, 14, 18],  # Low energy stations
            'constraints': {
                'min_energy_level': 50,
                'max_total_distance': 1400
            },
            'description': 'Energy balancing for low energy stations'
        }
    }
    return scenarios.get(scenario_type, None)

def validate_path(grid: EnergyGrid, path: List[int], constraints: Dict) -> bool:
    """
    Validate if a path meets all scenario constraints.
    
    Args:
        grid: The energy grid
        path: List of station IDs in the path
        constraints: Dictionary containing scenario-specific constraints
        
    Returns:
        bool: True if path is valid, False otherwise
    """
    if not path:
        return False
        
    # Check path continuity
    for i in range(len(path) - 1):
        if not grid.graph.has_edge(path[i], path[i + 1]):
            return False
            
    # Check total distance constraint
    total_distance = calculate_path_cost(grid, path)
    if 'max_total_distance' in constraints and total_distance > constraints['max_total_distance']:
        return False
        
    return True

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Energy Distribution Grid Pathfinding',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--scenario', 
        type=str, 
        choices=['emergency', 'maintenance', 'balancing'],
        required=True,
        help='''Select scenario to solve:
emergency   - Emergency response to critical station
maintenance - Planned maintenance route for 3 stations
balancing   - Energy balancing for low energy stations'''
    )
    
    args = parser.parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Create and display initial grid
        grid = EnergyGrid.create_fixed_grid()
        logger.info("Initializing Energy Distribution Grid...")
        visualizer = GridVisualizer()
        visualizer.display_initial_grid(grid)
        
        # Get scenario data
        scenario_data = get_scenario_data(args.scenario)
        if not scenario_data:
            raise ValueError(f"Invalid scenario selection: {args.scenario}")
        
        logger.info(f"Running {args.scenario} scenario: {scenario_data['description']}")
        
        # Solve the selected scenario
        path = None
        if args.scenario == 'emergency':
            path = solve_emergency_scenario(
                grid, 
                scenario_data['start'],
                scenario_data['targets'][0],
                scenario_data
            )

        elif args.scenario == 'maintenance':
            path = solve_maintenance_scenario(
                grid,
                scenario_data['start'],
                scenario_data['targets'],
                scenario_data
            )
        elif args.scenario == 'balancing':
            path = solve_balancing_scenario(
                grid,
                scenario_data['start'],
                scenario_data['targets'],
                scenario_data
            )
        
        if path:
            # Validate the solution
            if validate_path(grid, path, scenario_data['constraints']):
                # Calculate and log path cost
                total_cost = calculate_path_cost(grid, path)
                logger.info(f"Found optimal path! Total cost: {total_cost:.2f}")
                
                # Visualize the solution
                visualizer = GridVisualizer()
                visualizer.plot_scenario(
                    grid,
                    scenario_type=args.scenario,
                    path=path,
                    start_node=scenario_data['start'],
                    target_nodes=scenario_data['targets']
                )
                visualizer.show_plot()
            else:
                logger.error("Path found but doesn't meet required constraints!")
        else:
            logger.error("No valid solution found - Check your implementation!")
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())