import networkx as nx
from typing import Dict, List, Tuple, Optional
import random


class EnergyGrid:
    """Represents the energy distribution network."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.station_data = {}
        
    def add_station(self, station_id: int, 
                   position: Tuple[float, float],
                   energy_level: float = 100.0,
                   is_critical: bool = False,
                   status: str = "normal") -> None:
        """
        Add a station to the grid.
        
        Args:
            station_id: Unique identifier for the station
            position: (x, y) coordinates for visualization
            energy_level: Current energy level (0-100)
            is_critical: Whether the station is critical
            status: Station status ("normal", "critical", "low_energy")
        """
        self.graph.add_node(station_id)
        self.station_data[station_id] = {
            'pos': position,
            'energy_level': energy_level,
            'is_critical': is_critical,
            'status': status
        }
        
    def add_connection(self, station1: int, station2: int, 
                      weight: float) -> None:
        """
        Add a connection between stations with associated weight.
        
        Args:
            station1: ID of first station
            station2: ID of second station
            weight: Cost/distance of the connection
        """
        self.graph.add_edge(station1, station2, weight=weight)
        
    @classmethod
    def create_fixed_grid(cls) -> 'EnergyGrid':
        """
        Create a fixed grid with predefined stations and connections.
        
        Returns:
            EnergyGrid: A new grid with fixed layout
        """
        grid = cls()
        
        # Define fixed positions in a clear layout (roughly forming a city grid)
        positions = {
            # Central area (0-3)
            0: (300, 300),  # Central Station
            1: (200, 300),
            2: (400, 300),
            3: (300, 200),
            # North area (4-7)
            4: (200, 500),  # North Station
            5: (300, 500),
            6: (400, 500),
            7: (300, 400),
            # South area (8-11)
            8: (200, 100),
            9: (300, 100),  # South Station
            10: (400, 100),
            11: (300, 0),
            # East area (12-15)
            12: (500, 200),
            13: (500, 300),
            14: (500, 400),
            15: (600, 300),
            # West area (16-19)
            16: (100, 200),
            17: (100, 300),
            18: (100, 400),
            19: (0, 300),
        }
        
        # Define station statuses
        statuses = {
            # Critical stations
            3: {"status": "critical", "energy_level": 25, "is_critical": True},
            17: {"status": "critical", "energy_level": 18, "is_critical": True},
            
            # Low energy stations
            5: {"status": "low_energy", "energy_level": 22, "is_critical": False},
            10: {"status": "low_energy", "energy_level": 32, "is_critical": False},
            14: {"status": "low_energy", "energy_level": 28, "is_critical": False},
            18: {"status": "low_energy", "energy_level": 38, "is_critical": False},
            
            # Normal stations (will be applied to all others)
        }
        
        # Add all stations
        for station_id, pos in positions.items():
            status_data = statuses.get(station_id, {
                "status": "normal",
                "energy_level": 85,
                "is_critical": False
            })
            
            grid.add_station(
                station_id=station_id,
                position=pos,
                energy_level=status_data.get("energy_level", 90),
                is_critical=status_data.get("is_critical", False),
                status=status_data.get("status", "normal")
            )
        
        # Add connections with realistic distances
        connections = [
            # Central connections
            (0, 1, 120), (0, 2, 120), (0, 3, 120), (0, 7, 120),
            # North area
            (4, 5, 120), (5, 6, 120), (7, 5, 120),
            # South area
            (8, 9, 120), (9, 10, 120), (9, 11, 120),
            # East area
            (12, 13, 120), (13, 14, 120), (13, 15, 120),
            # West area
            (16, 17, 120), (17, 18, 120), (17, 19, 120),
            # Cross connections
            (1, 17, 120), (2, 13, 120), (3, 9, 120), (7, 14, 120),
            (4, 18, 180), (6, 14, 180), (8, 16, 180), (10, 12, 180)
        ]
        
        for start, end, weight in connections:
            grid.add_connection(start, end, weight)
            
        return grid
        
    def get_node_positions(self) -> Dict:
        """Return positions of all nodes for visualization."""
        return {node: self.station_data[node]['pos'] 
                for node in self.graph.nodes()}
        
    def get_node_colors(self) -> List:
        """Return colors for nodes based on their status."""
        color_map = {
            "normal": "lightblue",
            "critical": "red",
            "low_energy": "orange"
        }
        return [color_map[self.station_data[node]['status']] 
                for node in self.graph.nodes()]
        
    def get_edge_weights(self) -> Dict:
        """Return all edge weights."""
        return nx.get_edge_attributes(self.graph, 'weight') 
    

