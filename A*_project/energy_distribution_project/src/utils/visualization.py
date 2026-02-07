import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

class GridVisualizer:
    """Utility class for visualizing the energy distribution grid."""
    
    SCENARIO_CONFIGS = {
        'emergency': {
            'title': 'Emergency Scenario - Critical Station Response',
            'path_color': 'red',
            'description': (
                'Objective: Reach the most critical station in minimum time\n'
                'Start: Central Station (*)\n'
                'Target: Critical Station (!)\n'
                'Constraints: Minimize energy consumption'
            ),
            'start_marker': '*',
            'target_marker': '!'
        },
        'maintenance': {
            'title': 'Maintenance Scenario - Planned Station Visits',
            'path_color': 'blue',
            'description': (
                'Objective: Visit 3 stations in predefined order\n'
                'Start: North Station (*)\n'
                'Targets: Stations marked (1,2,3)\n'
                'Constraints: Optimize overall path length'
            ),
            'start_marker': '*',
            'target_marker': 'M'
        },
        'balancing': {
            'title': 'Energy Balancing Scenario - Low Energy Stations',
            'path_color': 'green',
            'description': (
                'Objective: Connect low energy stations efficiently\n'
                'Start: South Station (*)\n'
                'Targets: Low Energy Stations (E)\n'
                'Constraints: Minimize total distance'
            ),
            'start_marker': '*',
            'target_marker': 'E'
        }
    }
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """Initialize visualizer."""
        self.figsize = figsize
        plt.close('all')
    
    def display_initial_grid(self, grid) -> None:
        """
        Display the initial state of the grid showing all stations and their status.
        
        Args:
            grid: The energy grid instance to visualize
        """
        # Close all existing figures and create a new one
        plt.close('all')
        fig = plt.figure(figsize=self.figsize)
        
        # Get positions and colors
        pos = grid.get_node_positions()
        node_colors = grid.get_node_colors()
        
        # Draw the basic graph
        nx.draw_networkx_nodes(grid.graph, pos, 
                            node_color=node_colors,
                            node_size=1000)
        nx.draw_networkx_edges(grid.graph, pos, edge_color='gray')
        
        # Draw node labels with adjusted positioning
        labels = {node: f"S{node}\n{grid.station_data[node]['energy_level']:.0f}%" 
                for node in grid.graph.nodes()}
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0], coords[1])
        nx.draw_networkx_labels(grid.graph, pos_attrs, labels, font_size=9)
        
        # Draw edge weights
        edge_labels = {(u, v): f"{d:.1f}" 
                    for (u, v, d) in grid.graph.edges(data='weight')}
        nx.draw_networkx_edge_labels(grid.graph, pos, 
                                edge_labels=edge_labels,
                                font_size=8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='lightblue', markersize=10,
                    label='Normal Station (Energy > 50%)'),
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='red', markersize=10,
                    label='Critical Station (High Priority)'),
            plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor='orange', markersize=10,
                    label='Low Energy Station (< 50%)')
        ]
        
        plt.legend(handles=legend_elements, 
                loc='center left',
                bbox_to_anchor=(1, 0.5),
                title='Station Types',
                title_fontsize=12,
                fontsize=10)
        
        # Add title and information
        plt.suptitle('Energy Distribution Grid - Initial State', fontsize=14, y=0.95)
        plt.figtext(0.02, 0.02, 
                    'Station Format: Station ID\nEnergy Level %\n'
                    'Edge weights represent distance/cost units',
                    fontsize=10, style='italic',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.axis('off')
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        plt.close('all')
    
    def plot_scenario(self,
                     grid,
                     scenario_type: str,
                     path: Optional[List] = None,
                     start_node: Optional[int] = None,
                     target_nodes: Optional[List[int]] = None) -> None:
        """Plot a specific scenario with detailed information."""
        if scenario_type not in self.SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
            
        config = self.SCENARIO_CONFIGS[scenario_type]
        
        # Close all existing figures and create a new one
        plt.close('all')
        fig = plt.figure(figsize=self.figsize)
        
        # Get positions and colors
        pos = grid.get_node_positions()
        node_colors = grid.get_node_colors()
        
        # Draw the basic graph
        nx.draw_networkx_nodes(grid.graph, pos, 
                             node_color=node_colors,
                             node_size=1000)
        nx.draw_networkx_edges(grid.graph, pos, edge_color='gray')
        
        # Draw start node marker
        if start_node is not None:
            plt.scatter(pos[start_node][0], pos[start_node][1], 
                      c='darkgreen', marker=config['start_marker'], 
                      s=500, zorder=5)
        
        # Draw target nodes markers
        if target_nodes:
            for idx, node in enumerate(target_nodes, 1):
                if scenario_type == 'maintenance':
                    marker_text = str(idx)
                else:
                    marker_text = config['target_marker']
                
                plt.scatter(pos[node][0], pos[node][1], 
                          c='darkred', marker='s' if scenario_type != 'maintenance' else 'o',
                          s=300, zorder=5)
                plt.annotate(marker_text,
                            xy=pos[node],
                            xytext=(0, 0),
                            textcoords='offset points',
                            ha='center',
                            va='center',
                            color='white',
                            fontweight='bold',
                            fontsize=10,
                            zorder=6)
        
        # Draw node labels with adjusted positioning
        labels = {node: f"S{node}" for node in grid.graph.nodes()}
        pos_attrs = {}
        for node, coords in pos.items():
            pos_attrs[node] = (coords[0], coords[1] + 20)
        nx.draw_networkx_labels(grid.graph, pos_attrs, labels, font_size=10)
        
        # Draw edge weights
        edge_labels = {(u, v): f"{d:.1f}" 
                      for (u, v, d) in grid.graph.edges(data='weight')}
        nx.draw_networkx_edge_labels(grid.graph, pos, 
                                   edge_labels=edge_labels,
                                   font_size=8)
        
        # Highlight path if provided
        if path and len(path) > 1:
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(grid.graph, pos,
                                 edgelist=path_edges,
                                 edge_color=config['path_color'],
                                 width=3)
        
        # Add detailed legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='lightblue', markersize=10,
                      label='Normal Station (Energy > 50%)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='red', markersize=10,
                      label='Critical Station (Priority)'),
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='orange', markersize=10,
                      label='Low Energy Station (< 50%)'),
            plt.Line2D([0], [0], color=config['path_color'],
                      linewidth=3, label='Optimized Path'),
            plt.Line2D([0], [0], marker='*', color='darkgreen',
                      markersize=15, label='Start Station'),
        ]
        
        if scenario_type == 'maintenance':
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='darkred', markersize=10,
                          label='Visit Order (1,2,3)'))
        else:
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w',
                          markerfacecolor='darkred', markersize=10,
                          label='Target Station'))
        
        plt.legend(handles=legend_elements, 
                  loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  title='Legend',
                  title_fontsize=12,
                  fontsize=10)
        
        # Add title and description
        plt.suptitle(config['title'], fontsize=14, y=0.95)
        plt.figtext(0.02, 0.02, config['description'], 
                   fontsize=10, style='italic',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Add edge weight explanation
        plt.figtext(0.02, 0.95, 
                   'Edge weights represent distance/cost units',
                   fontsize=10, style='italic')
        
        plt.axis('off')
        plt.tight_layout()
    
    def save_plot(self, filename: str) -> None:
        """Save the current plot to a file."""
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        
    def show_plot(self) -> None:
        """Display the current plot."""
        plt.show()