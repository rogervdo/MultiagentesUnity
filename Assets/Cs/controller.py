import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import heapq
import socket
import json

# Optional heuristics import
try:
    from traffic_heuristics import HeuristicsMixin

    HEURISTICS_AVAILABLE = True  # Temporarily disabled for testing
except ImportError:
    HEURISTICS_AVAILABLE = False
    print("Warning: traffic_heuristics module not found. Running without heuristics.")

from puntos import (
    POSITIONS,
    NODES_LIST,
    CONNECTIONS,
    STOPLIGHT_NODES,
    GROUP_NODES,
    TRAFFIC_PRESETS,
)

from legend import LEGEND_ELEMENTS

"""
Traffic Simulation with Unity Integration

JSON Data Structure:
{
    "time": simulation_step,
    "active_group": current_traffic_light_group,
    "cars": [
        {
            "id": car_index,
            "x": position_x,
            "y": position_y,
            "state": "moving"|"arrived",
            "spawn_node": starting_node_id,
            "target_node": destination_node_id
        }
    ]
}
"""
# Simulation parameters
params = {
    "steps": 100,
    "spawn_rate": 1,  # Probability of spawning a car each step
    "world_size": 500,  # World boundaries (0 to world_size)
    "preset": "morning",  # Traffic preset: "morning", "evening", "night"
    "animation": True,  # Save to gif
}


class TrafficNode:
    """Represents a node in the road network"""

    def __init__(self, node_id, x, y, node_type="intersection", group_id=None):
        self.id = node_id
        self.x = x
        self.y = y
        self.node_type = node_type  # "spawn", "intersection", "destination"
        self.group_id = (
            group_id  # Group ID for stoplight functionality (None if not applicable)
        )
        self.connected_nodes = {}  # {node_id: distance}

    def add_connection(self, target_node_id, distance):
        """Add a connection to another node"""
        self.connected_nodes[target_node_id] = distance

    def get_position(self):
        return np.array([self.x, self.y])


class Car(ap.Agent):
    """Car agent that moves between nodes"""

    def setup(self, start_node_id, target_node_id):
        self.current_node_id = start_node_id
        self.target_node_id = target_node_id
        self.path = []  # Will store the path as list of node IDs
        self.current_path_index = 0
        self.state = "moving"  # 'moving', 'arrived'

        # Add waiting steps tracking
        self.waiting_steps = 0  # Track how many steps this car has been waiting

        # Assign color based on spawn node
        spawn_colors = {
            "8_26": "#4169E1",  # Royal blue
            "18_2": "#32CD32",  # Lime green
            "2_14": "#FF9500",  # Deep pink
            "2_12": "#FFD700",  # Gold
            "22_4": "#9932CC",  # Dark orchid
            "30_15": "#FF4500",  # Orange red
        }
        self.color = spawn_colors.get(start_node_id, "white")

        # Calculate initial path
        self.calculate_path()

        # Set initial position
        if self.path:
            start_node = self.model.road_network.nodes[self.current_node_id]
            self.position = start_node.get_position().copy()
            # Set occupancy
            self.model.node_occupancy[self.current_node_id] = self.id
        else:
            self.state = "arrived"

    def calculate_path(self):
        """Pathfinding using A* algorithm for optimal paths"""
        network = self.model.road_network

        # If direct connection exists, use it
        if self.target_node_id in network.nodes[self.current_node_id].connected_nodes:
            self.path = [self.current_node_id, self.target_node_id]
        else:
            # Use A* pathfinding for optimal path
            self.path = self.find_astar_path()

    def find_astar_path(self):
        """Find optimal path using A* algorithm"""
        network = self.model.road_network
        start = self.current_node_id
        goal = self.target_node_id

        # Priority queue for open set: (f_score, node_id)
        open_set = []
        heapq.heappush(open_set, (0, start))

        # Came from dictionary to reconstruct path
        came_from = {}

        # g_score: cost from start to current node
        g_score = {node_id: float("inf") for node_id in network.nodes}
        g_score[start] = 0

        # f_score: estimated total cost from start to goal through current node
        f_score = {node_id: float("inf") for node_id in network.nodes}
        f_score[start] = self.heuristic(start, goal)

        while open_set:
            # Get node with lowest f_score
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                return self.reconstruct_path(came_from, current)

            # Check all neighbors
            for neighbor in network.nodes[current].connected_nodes:
                # Calculate tentative g_score
                edge_distance = network.nodes[current].connected_nodes[neighbor]
                tentative_g_score = g_score[current] + edge_distance

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal
                    )

                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return []

    def heuristic(self, node_id1, node_id2):
        """Euclidean distance heuristic"""
        network = self.model.road_network
        pos1 = network.nodes[node_id1].get_position()
        pos2 = network.nodes[node_id2].get_position()
        return np.linalg.norm(pos2 - pos1)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_alternative_path(self, blocked_node_id):
        """Find alternative path avoiding a blocked node"""
        network = self.model.road_network
        start = self.current_node_id
        goal = self.target_node_id

        # Priority queue for open set: (f_score, node_id)
        open_set = []
        heapq.heappush(open_set, (0, start))

        # Came from dictionary to reconstruct path
        came_from = {}

        # g_score: cost from start to current node
        g_score = {node_id: float("inf") for node_id in network.nodes}
        g_score[start] = 0

        # f_score: estimated total cost from start to goal through current node
        f_score = {node_id: float("inf") for node_id in network.nodes}
        f_score[start] = self.heuristic(start, goal)

        while open_set:
            # Get node with lowest f_score
            current_f, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                return self.reconstruct_path(came_from, current)

            # Check all neighbors (excluding the blocked node)
            for neighbor in network.nodes[current].connected_nodes:
                if neighbor == blocked_node_id:
                    continue  # Skip the blocked node

                # Calculate tentative g_score
                edge_distance = network.nodes[current].connected_nodes[neighbor]
                tentative_g_score = g_score[current] + edge_distance

                if tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal
                    )

                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return []

    def step(self):
        if self.state == "arrived" or not self.path:
            return

        # Check if we've reached the end of our path
        if self.current_path_index >= len(self.path) - 1:
            self.state = "arrived"
            # Immediately clear occupancy when arriving at destination
            self.model.node_occupancy[self.current_node_id] = None
            return

        # Get next node
        next_node_id = self.path[self.current_path_index + 1]
        next_node = self.model.road_network.nodes[next_node_id]

        # Check if next node is occupied
        if self.model.node_occupancy[next_node_id] is not None:
            # Check if we're currently on a traffic light node
            if self.current_node_id in STOPLIGHT_NODES:
                # Try to find an alternative route avoiding the blocked node
                alternative_path = self.find_alternative_path(next_node_id)
                if alternative_path and len(alternative_path) > 1:
                    print(
                        f"Car {self.id} found alternative route around blocked node {next_node_id}"
                    )
                    self.path = alternative_path
                    self.current_path_index = 0
                    # Recalculate next node after path change
                    next_node_id = self.path[self.current_path_index + 1]
                    next_node = self.model.road_network.nodes[next_node_id]
                else:
                    # Cannot move - increment waiting steps
                    self.waiting_steps += 1
                    return  # No alternative route found, wait
            else:
                # Cannot move - increment waiting steps
                self.waiting_steps += 1
                return  # Not on traffic light, wait for node to be free

        # Yield check for merge points
        for rule in self.model.yield_rules:
            if self.current_node_id == rule["yield"] and next_node_id == rule["merge"]:
                if self.model.node_occupancy[rule["priority"]] is not None:
                    # Cannot move - increment waiting steps
                    self.waiting_steps += 1
                    return  # Yield to priority car

        # Group check for stoplight functionality
        current_node = self.model.road_network.nodes[self.current_node_id]
        if current_node.group_id is not None:
            if next_node_id in STOPLIGHT_NODES:
                if current_node.group_id != self.model.active_group:
                    # Cannot move - increment waiting steps
                    self.waiting_steps += 1
                    return  # Wait for green light

                # Additional congestion check - even if it's our turn, check for congestion ahead
                if self.model.check_congestion_ahead(
                    self.current_node_id, next_node_id
                ):
                    # Cannot move - increment waiting steps
                    self.waiting_steps += 1
                    return  # Wait due to congestion ahead

        # Move to next node instantly
        self.position = next_node.get_position().copy()
        # Update occupancy
        self.model.node_occupancy[self.current_node_id] = None
        self.model.node_occupancy[next_node_id] = self.id
        # Reset wait time for the node we're leaving (car successfully moved)
        self.model.node_wait_times[self.current_node_id] = 0
        self.current_node_id = next_node_id
        self.current_path_index += 1


class RoadNetwork:
    """Manages the road network of nodes and connections"""

    def __init__(self):
        self.nodes = {}
        self.spawn_nodes = []
        self.destination_nodes = []

    def add_node(self, node_id, x, y, node_type="intersection", group_id=None):
        """Add a node to the network"""
        node = TrafficNode(node_id, x, y, node_type, group_id)
        self.nodes[node_id] = node

        if node_type == "spawn":
            self.spawn_nodes.append(node_id)
        elif node_type == "destination":
            self.destination_nodes.append(node_id)

    def create_simple_network(self, world_size):
        # Additional new path nodes (user request)

        # New path nodes (user request)
        """Create custom road network based on specified layout"""
        # Manual POSITIONS for nodes, independent of x,y calculations
        for node_id, node_type, group_id in NODES_LIST:
            x, y = POSITIONS[node_id]
            self.add_node(node_id, x, y, node_type, group_id)

        # Add all connections as directed edges
        for node1, node2 in CONNECTIONS:
            if node1 in self.nodes and node2 in self.nodes:
                pos1 = self.nodes[node1].get_position()
                pos2 = self.nodes[node2].get_position()
                distance = np.linalg.norm(pos2 - pos1)
                # Add as directed edge (one-way)
                self.nodes[node1].add_connection(node2, distance)


class TrafficModelBase(ap.Model):
    """Main simulation model"""

    def setup(self):
        # Create road network
        self.road_network = RoadNetwork()
        self.road_network.create_simple_network(self.p["world_size"])

        # Initialize node occupancy
        self.node_occupancy = {node_id: None for node_id in self.road_network.nodes}

        # Track how long cars have been waiting at each node (for congestion detection)
        self.node_wait_times = {node_id: 0 for node_id in self.road_network.nodes}

        # Initialize car list
        self.cars = ap.AgentList(self, 0, Car)

        # Statistics
        self.total_cars_spawned = 0
        self.total_cars_arrived = 0

        # Waiting steps tracking
        self.avg_waiting_steps_history = []  # Track average waiting steps over time

        # Traffic light state
        self.active_group = 0  # Start with group 0
        self.group_cycle_steps = 5  # Switch every 5 steps
        self.traffic_light_counter = 0  # Counter for regular traffic light changes

        # Yield rules for merge points
        self.yield_rules = [
            {"yield": "16_21", "merge": "18_22", "priority": "18_20"},
            {"yield": "5_17", "merge": "4_16", "priority": "6_16"},
            {"yield": "22_10", "merge": "22_12", "priority": "20_12"},
            {"yield": "10_7", "merge": "8_6", "priority": "8_8"},
        ]

        self.group_nodes = GROUP_NODES
        self.traffic_presets = TRAFFIC_PRESETS

        # Set the active preset based on parameters
        preset_name = self.p.get("preset", "morning")
        if preset_name not in self.traffic_presets:
            print(
                f"Warning: Preset '{preset_name}' not found, using 'morning' as default"
            )
            preset_name = "morning"

        self.spawn_destinations = self.traffic_presets[preset_name]
        print(f"Using traffic preset: {preset_name}")

    def step(self):
        # Increment traffic light counter for regular timing
        self.traffic_light_counter += 1

        # Handle traffic light group changes differently based on heuristics availability
        if not HEURISTICS_AVAILABLE:
            # No heuristics: Change groups every 5 steps (fixed timer)
            if self.traffic_light_counter % 5 == 0:
                self.active_group = (self.active_group + 1) % 3
                print(f"Traffic light switched to Group {self.active_group}")
        else:
            # Heuristics available: Use adaptive timer controlled by heuristics
            if self.traffic_light_counter % self.group_cycle_steps == 0:
                self.active_group = (self.active_group + 1) % 3
                print(f"Traffic light switched to Group {self.active_group}")
                # Adjust timer based on heuristics
                if hasattr(self, "adjust_timer_based_on_traffic"):
                    self.adjust_timer_based_on_traffic()

        # Spawn new cars with restricted destinations
        if (
            random.random() < self.p["spawn_rate"]
            and self.road_network.spawn_nodes
            and self.road_network.destination_nodes
        ):
            # Calculate spawn weights based on total route capacity for each spawn point
            spawn_weights = []
            valid_spawn_nodes = []

            for spawn_node in self.road_network.spawn_nodes:
                dest_probs = self.spawn_destinations.get(spawn_node, {})
                # Filter to only include destinations that exist in the network
                available_dests = {
                    dest: prob
                    for dest, prob in dest_probs.items()
                    if dest in self.road_network.destination_nodes
                }

                if available_dests:
                    # Use sum of route weights as the spawn weight for this spawn point
                    total_weight = sum(available_dests.values())
                    spawn_weights.append(total_weight)
                    valid_spawn_nodes.append(spawn_node)

            if valid_spawn_nodes:
                # Select spawn point weighted by total route capacity
                spawn_node = random.choices(
                    valid_spawn_nodes, weights=spawn_weights, k=1
                )[0]

                # Get destination probabilities for the selected spawn point
                dest_probs = self.spawn_destinations.get(spawn_node, {})
                available_dests = {
                    dest: prob
                    for dest, prob in dest_probs.items()
                    if dest in self.road_network.destination_nodes
                }

                if available_dests:
                    destinations = list(available_dests.keys())
                    probabilities = list(available_dests.values())
                    dest_node = random.choices(
                        destinations, weights=probabilities, k=1
                    )[0]
                    new_car = Car(
                        self, start_node_id=spawn_node, target_node_id=dest_node
                    )
                    self.cars.append(new_car)
                    self.total_cars_spawned += 1

                    # Debug output to show car assignments
                    if self.t % 50 == 0:  # Print every 50 steps to avoid spam
                        print(
                            f"Car {self.total_cars_spawned} spawned: {spawn_node} → {dest_node}"
                        )

        # Update all cars
        arrived_cars = []
        for car in self.cars:
            car.step()
            if car.state == "arrived":
                arrived_cars.append(car)

        # Remove arrived cars and log their completion
        for car in arrived_cars:
            print(
                f"Car completed journey: {car.path[0] if car.path else 'unknown'} → {car.target_node_id}"
            )
            # Clear occupancy (safety check - should already be cleared when car arrived)
            if self.node_occupancy[car.current_node_id] == car.id:
                self.node_occupancy[car.current_node_id] = None
                # Reset wait time when car arrives at destination
                self.node_wait_times[car.current_node_id] = 0
            self.cars.remove(car)
            self.total_cars_arrived += 1

        # Update node wait times for congestion detection
        for node_id in self.node_occupancy:
            if self.node_occupancy[node_id] is not None:
                # Node is occupied, increment wait time
                self.node_wait_times[node_id] += 1
            else:
                # Node is free, reset wait time
                self.node_wait_times[node_id] = 0

        # Calculate average waiting steps across all active cars
        if len(self.cars) > 0:
            total_waiting_steps = sum(car.waiting_steps for car in self.cars)
            avg_waiting_steps = total_waiting_steps / len(self.cars)
        else:
            avg_waiting_steps = 0.0

        self.avg_waiting_steps_history.append(avg_waiting_steps)

        # Print statistics every 100 steps
        if self.t % 100 == 0 and self.t > 0:
            print(
                f"Step {self.t}: {len(self.cars)} active cars, "
                f"{self.total_cars_spawned} spawned, "
                f"{self.total_cars_arrived} arrived, "
                f"Avg waiting steps: {avg_waiting_steps:.2f}"
            )

    def check_congestion_ahead(self, current_node_id, next_node_id):
        """Check if specified nodes ahead are occupied"""
        # Define specific congestion check mappings: current_node -> nodes_to_check
        # Each traffic light node checks specific downstream nodes for congestion
        congestion_check_nodes = {
            # Group 1 traffic lights - check roundabout entries and other intersections
            "6_14": ["12_12", "16_13", "8_14"],  # 8_14 checks both roundabout entries
            "6_12": ["10_11", "16_11", "8_12"],  # 8_12 checks both roundabout entries
            # Additional mappings for better congestion control
            "20_15": ["16_15", "16_17"],  # Roundabout internal checks
        }

        # Get the nodes to check for this specific current node
        check_nodes = congestion_check_nodes.get(current_node_id, [])

        # If no specific mapping found, try group-specific mappings for nodes that appear in multiple groups
        if not check_nodes:
            current_node = self.road_network.nodes.get(current_node_id)
            if current_node and current_node.group_id is not None:
                group_specific_mappings = {
                    ("8_16", 2): ["10_15"],  # 8_16 in group 2
                    ("18_16", 2): ["10_17"],  # 18_16 in group 2
                }
                check_nodes = group_specific_mappings.get(
                    (current_node_id, current_node.group_id), []
                )

        # Check if ALL of the critical nodes are occupied
        if check_nodes:
            all_occupied = True
            for node_id in check_nodes:
                if self.node_occupancy.get(node_id) is None:
                    all_occupied = False
                    break

            if all_occupied:
                return True  # Congestion detected - ALL nodes occupied

        return False  # No congestion


# Function to create TrafficModel with optional heuristics
def create_traffic_model_class(use_heuristics=True):
    """Create TrafficModel class with or without heuristics"""
    if use_heuristics and HEURISTICS_AVAILABLE:

        class TrafficModelWithHeuristics(TrafficModelBase, HeuristicsMixin):
            pass

        return TrafficModelWithHeuristics
    else:
        return TrafficModelBase


# Create the TrafficModel class
TrafficModel = create_traffic_model_class(use_heuristics=HEURISTICS_AVAILABLE)


def draw_simulation(model, ax):
    """Draw the current state of the simulation"""
    ax.clear()
    # Set axis limits to show all nodes with some padding
    ax.set_xlim(-200, 450)
    ax.set_ylim(-150, 450)
    ax.set_aspect("equal")
    ax.set_title(f"Node-Based Traffic Simulation - Step {model.t}")

    # Draw network nodes
    group_colors = {0: "orange", 1: "purple", 2: "cyan"}
    for node_id, node in model.road_network.nodes.items():
        if node_id in STOPLIGHT_NODES:
            # Highlight stoplight nodes with active group color
            color = group_colors.get(model.active_group, "yellow")
        elif node.group_id is not None:
            # Color based on group_id
            color = group_colors.get(
                node.group_id, "yellow"
            )  # Default to yellow if unknown group
        else:
            color = (
                "green"
                if node.node_type == "spawn"
                else "red"
                if node.node_type == "destination"
                else "blue"
            )

        ax.add_patch(Circle((node.x, node.y), 8, color=color, alpha=0.7))
        ax.text(node.x, node.y - 15, node_id, ha="center", fontsize=8)

    # Draw network edges as arrows
    yield_connections = [
        ("16_21", "18_22"),
        ("5_17", "4_16"),
        ("22_10", "22_12"),
        ("10_7", "8_6"),
    ]
    for node_id, node in model.road_network.nodes.items():
        for connected_id in node.connected_nodes:
            connected_node = model.road_network.nodes[connected_id]
            dx = connected_node.x - node.x
            dy = connected_node.y - node.y
            # Special color for yield connections
            if (node_id, connected_id) in yield_connections:
                arrow_color = "red"
            else:
                arrow_color = "k"
            ax.arrow(
                node.x,
                node.y,
                dx,
                dy,
                head_width=8,
                head_length=12,
                fc=arrow_color,
                ec=arrow_color,
                alpha=0.3,
                length_includes_head=True,
            )

    # Draw cars with spawn-based colors
    if len(model.cars) > 0:
        car_positions = np.array([car.position for car in model.cars])
        car_colors = [car.color for car in model.cars]

        ax.scatter(
            car_positions[:, 0],
            car_positions[:, 1],
            c=car_colors,
            s=40,
            zorder=5,
            edgecolors="black",
            linewidth=0.5,
        )

    # Add legend
    ax.legend(handles=LEGEND_ELEMENTS, loc="upper right")


def run_simulation_with_animation(filename="H6-Traffic.gif"):
    """Run the simulation with animation"""
    model = TrafficModel(params)

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))

    def animate_step(model, ax):
        draw_simulation(model, ax)

    # Run with animation
    animation = ap.animate(model, fig, ax, animate_step)

    # Save animation with specified filename
    if params["animation"]:
        animation.save(filename, writer="pillow", fps=5)
        print(f"Animation saved as '{filename}'")

    return model, animation


class TCPSender:
    def __init__(self, host="localhost", port=1101):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to Unity at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def send_complete_json(self, json_file_path):
        if not self.connected:
            return False
        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)

            json_str = json.dumps(data)
            size = len(json_str.encode("utf-8"))

            # Send size first (4 bytes)
            self.socket.sendall(size.to_bytes(4, byteorder="big"))
            # Then send the JSON data
            self.socket.sendall(json_str.encode("utf-8"))
            print(f"Sent complete JSON ({size} bytes)")
            return True
        except Exception as e:
            print(f"Send complete JSON failed: {e}")
            return False

    def disconnect(self):
        if self.socket:
            self.socket.close()
        self.connected = False


def send_json_file_to_unity(json_file_path, host="localhost", port=1101):
    sender = TCPSender(host, port)
    if sender.connect():
        sender.send_complete_json(json_file_path)
        sender.disconnect()


def run_multiple_simulations_and_plot(num_runs=10, steps=None):
    """
    Run multiple simulations comparing heuristics vs no heuristics and plot comparative statistics

    Args:
        num_runs: Number of simulation runs to perform for each mode
        steps: Number of steps per simulation (uses params["steps"] if None)
    """
    if steps is None:
        steps = params["steps"]

    print(
        f"Running comparative analysis: {num_runs} simulations with {steps} steps each..."
    )

    def run_simulation_set(use_heuristics, description):
        """Run a set of simulations with specified heuristics setting"""
        print(f"\nRunning {description}...")

        # Store data from all runs
        all_active_cars = []
        all_cumulative_arrivals = []
        all_avg_waiting_steps = []

        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...")

            # Create model with specified heuristics setting
            TrafficModelClass = create_traffic_model_class(
                use_heuristics=use_heuristics
            )
            model = TrafficModelClass(params)
            model.setup()

            active_cars_data = []
            cumulative_arrivals_data = []
            avg_waiting_steps_data = []

            for t in range(steps):
                model.t = t + 1
                model.step()

                # Collect data
                active_cars = len(model.cars)
                cumulative_arrivals = model.total_cars_arrived
                avg_waiting_steps = (
                    model.avg_waiting_steps_history[-1]
                    if model.avg_waiting_steps_history
                    else 0.0
                )

                active_cars_data.append(active_cars)
                cumulative_arrivals_data.append(cumulative_arrivals)
                avg_waiting_steps_data.append(avg_waiting_steps)

            all_active_cars.append(active_cars_data)
            all_cumulative_arrivals.append(cumulative_arrivals_data)
            all_avg_waiting_steps.append(avg_waiting_steps_data)

        return {
            "active_cars": np.array(all_active_cars),
            "cumulative_arrivals": np.array(all_cumulative_arrivals),
            "avg_waiting_steps": np.array(all_avg_waiting_steps),
        }

    # Run simulations with and without heuristics
    heuristics_data = run_simulation_set(
        use_heuristics=True, description="simulations WITH heuristics"
    )
    no_heuristics_data = run_simulation_set(
        use_heuristics=False, description="simulations WITHOUT heuristics"
    )

    # Calculate averages and standard deviations
    h_avg_active = np.mean(heuristics_data["active_cars"], axis=0)
    h_std_active = np.std(heuristics_data["active_cars"], axis=0)
    h_avg_arrivals = np.mean(heuristics_data["cumulative_arrivals"], axis=0)
    h_std_arrivals = np.std(heuristics_data["cumulative_arrivals"], axis=0)
    h_avg_waiting = np.mean(heuristics_data["avg_waiting_steps"], axis=0)
    h_std_waiting = np.std(heuristics_data["avg_waiting_steps"], axis=0)

    nh_avg_active = np.mean(no_heuristics_data["active_cars"], axis=0)
    nh_std_active = np.std(no_heuristics_data["active_cars"], axis=0)
    nh_avg_arrivals = np.mean(no_heuristics_data["cumulative_arrivals"], axis=0)
    nh_std_arrivals = np.std(no_heuristics_data["cumulative_arrivals"], axis=0)
    nh_avg_waiting = np.mean(no_heuristics_data["avg_waiting_steps"], axis=0)
    nh_std_waiting = np.std(no_heuristics_data["avg_waiting_steps"], axis=0)

    # Create comparative plots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 12))

    time_steps = range(1, steps + 1)

    # Plot 1: Active cars comparison
    ax1.plot(time_steps, h_avg_active, "b-", linewidth=2, label="With Heuristics")
    ax1.fill_between(
        time_steps,
        h_avg_active - h_std_active,
        h_avg_active + h_std_active,
        alpha=0.3,
        color="blue",
    )
    ax1.plot(time_steps, nh_avg_active, "r-", linewidth=2, label="Without Heuristics")
    ax1.fill_between(
        time_steps,
        nh_avg_active - nh_std_active,
        nh_avg_active + nh_std_active,
        alpha=0.3,
        color="red",
    )
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Number of Active Cars")
    ax1.set_title("Active Cars Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Cumulative arrivals comparison
    ax2.plot(time_steps, h_avg_arrivals, "b-", linewidth=2, label="With Heuristics")
    ax2.fill_between(
        time_steps,
        h_avg_arrivals - h_std_arrivals,
        h_avg_arrivals + h_std_arrivals,
        alpha=0.3,
        color="blue",
    )
    ax2.plot(time_steps, nh_avg_arrivals, "r-", linewidth=2, label="Without Heuristics")
    ax2.fill_between(
        time_steps,
        nh_avg_arrivals - nh_std_arrivals,
        nh_avg_arrivals + nh_std_arrivals,
        alpha=0.3,
        color="red",
    )
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Cumulative Car Arrivals")
    ax2.set_title("Cumulative Car Arrivals Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Average waiting steps comparison
    ax3.plot(time_steps, h_avg_waiting, "b-", linewidth=2, label="With Heuristics")
    ax3.fill_between(
        time_steps,
        h_avg_waiting - h_std_waiting,
        h_avg_waiting + h_std_waiting,
        alpha=0.3,
        color="blue",
    )
    ax3.plot(time_steps, nh_avg_waiting, "r-", linewidth=2, label="Without Heuristics")
    ax3.fill_between(
        time_steps,
        nh_avg_waiting - nh_std_waiting,
        nh_avg_waiting + nh_std_waiting,
        alpha=0.3,
        color="red",
    )
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Average Waiting Steps")
    ax3.set_title("Average Waiting Steps Over Time")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Difference in active cars
    diff_active = h_avg_active - nh_avg_active
    ax4.plot(
        time_steps, diff_active, "g-", linewidth=2, label="Heuristics - No Heuristics"
    )
    ax4.fill_between(
        time_steps,
        diff_active - (h_std_active + nh_std_active),
        diff_active + (h_std_active + nh_std_active),
        alpha=0.3,
        color="green",
    )
    ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Difference in Active Cars")
    ax4.set_title("Difference: Heuristics vs No Heuristics\n(Active Cars)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Difference in cumulative arrivals
    diff_arrivals = h_avg_arrivals - nh_avg_arrivals
    ax5.plot(
        time_steps,
        diff_arrivals,
        "purple",
        linewidth=2,
        label="Heuristics - No Heuristics",
    )
    ax5.fill_between(
        time_steps,
        diff_arrivals - (h_std_arrivals + nh_std_arrivals),
        diff_arrivals + (h_std_arrivals + nh_std_arrivals),
        alpha=0.3,
        color="purple",
    )
    ax5.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Time Step")
    ax5.set_ylabel("Difference in Cumulative Arrivals")
    ax5.set_title("Difference: Heuristics vs No Heuristics\n(Cumulative Arrivals)")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Difference in average waiting steps
    diff_waiting = h_avg_waiting - nh_avg_waiting
    ax6.plot(
        time_steps,
        diff_waiting,
        "orange",
        linewidth=2,
        label="Heuristics - No Heuristics",
    )
    ax6.fill_between(
        time_steps,
        diff_waiting - (h_std_waiting + nh_std_waiting),
        diff_waiting + (h_std_waiting + nh_std_waiting),
        alpha=0.3,
        color="orange",
    )
    ax6.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax6.set_xlabel("Time Step")
    ax6.set_ylabel("Difference in Average Waiting Steps")
    ax6.set_title("Difference: Heuristics vs No Heuristics\n(Average Waiting Steps)")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()
    plt.savefig("traffic_heuristics_comparison.png", dpi=300, bbox_inches="tight")
    print("Comparative plot saved as 'traffic_heuristics_comparison.png'")

    # Don't show plot in headless environments
    try:
        plt.show()
    except Exception:
        print("Plot display not available in headless environment")

    # Print comparative summary statistics
    print("\n" + "=" * 80)
    print("HEURISTICS VS NO HEURISTICS COMPARISON")
    print("=" * 80)
    print(f"Number of runs per mode: {num_runs}")
    print(f"Steps per simulation: {steps}")
    print()

    print("FINAL STATISTICS:")
    print(f"Final active cars with heuristics: {h_avg_active[-1]:.2f}")
    print(f"Final active cars without heuristics: {nh_avg_active[-1]:.2f}")
    print(f"Final cumulative arrivals with heuristics: {h_avg_arrivals[-1]:.2f}")
    print(f"Final cumulative arrivals without heuristics: {nh_avg_arrivals[-1]:.2f}")
    print(f"Final average waiting steps with heuristics: {h_avg_waiting[-1]:.2f}")
    print(f"Final average waiting steps without heuristics: {nh_avg_waiting[-1]:.2f}")
    print()

    print("PERFORMANCE METRICS:")
    final_active_diff = h_avg_active[-1] - nh_avg_active[-1]
    final_arrival_diff = h_avg_arrivals[-1] - nh_avg_arrivals[-1]
    final_waiting_diff = h_avg_waiting[-1] - nh_avg_waiting[-1]
    efficiency_h = h_avg_arrivals[-1] / (h_avg_arrivals[-1] + h_avg_active[-1]) * 100
    efficiency_nh = (
        nh_avg_arrivals[-1] / (nh_avg_arrivals[-1] + nh_avg_active[-1]) * 100
    )

    print(f"Active cars difference (H - NoH): {final_active_diff:.2f}")
    print(f"Cumulative arrivals difference (H - NoH): {final_arrival_diff:.2f}")
    print(f"Average waiting steps difference (H - NoH): {final_waiting_diff:.2f}")
    print(f"Efficiency with heuristics: {efficiency_h:.2f}%")
    print(f"Efficiency without heuristics: {efficiency_nh:.2f}%")
    print(f"Efficiency improvement: {efficiency_h - efficiency_nh:.2f}%")

    return {
        "heuristics": {
            "avg_active_cars": h_avg_active,
            "avg_cumulative_arrivals": h_avg_arrivals,
            "avg_waiting_steps": h_avg_waiting,
            "std_active_cars": h_std_active,
            "std_cumulative_arrivals": h_std_arrivals,
            "std_waiting_steps": h_std_waiting,
        },
        "no_heuristics": {
            "avg_active_cars": nh_avg_active,
            "avg_cumulative_arrivals": nh_avg_arrivals,
            "avg_waiting_steps": nh_avg_waiting,
            "std_active_cars": nh_std_active,
            "std_cumulative_arrivals": nh_std_arrivals,
            "std_waiting_steps": nh_std_waiting,
        },
    }


def run_simulation_and_send_json():
    frames = []
    model = TrafficModel(params)
    model.setup()
    for t in range(params["steps"]):
        model.t = t + 1  # Manually set time counter for traffic light cycling
        model.step()
        frame = {
            "time": model.t,
            "active_group": model.active_group,  # Traffic light group instead of lights
            "cars": [
                {
                    "id": car.id,  # Use persistent car ID from agentpy
                    "x": float(car.position[0]),
                    "y": float(car.position[1]),
                    "state": car.state,
                    "current_node": car.current_node_id,
                    "spawn_node": car.path[0] if car.path else "unknown",
                    "target_node": car.target_node_id,
                    "is_at_exit": car.current_node_id
                    in model.road_network.destination_nodes,
                }
                for i, car in enumerate(model.cars)
            ],
        }
        frames.append(frame)

    # Guardar
    with open("data.json", "w") as f:
        json.dump(frames, f)

    # Mandar a unity
    send_json_file_to_unity("data.json")
    print("ENVIADO A UNITY EXITOSAMENTE!!!!!! YAY")


if __name__ == "__main__":
    print("Traffic Simulation with Unity Integration")
    print("=" * 50)
    print("Choose mode:")
    print("1. Single simulation with Unity export")
    print("2. Multiple runs with statistics and plots")
    try:
        mode_choice = input("Enter mode (1 or 2, default: 1): ").strip()
    except EOFError:
        mode_choice = "2"  # Default to multiple runs mode for testing

    # Get preset choice
    preset_choice = input("Choose traffic preset (1-3, default: 1): ").strip()
    if preset_choice == "1" or preset_choice == "":
        params["preset"] = "morning"
    elif preset_choice == "2":
        params["preset"] = "evening"
    elif preset_choice == "3":
        params["preset"] = "night"
    else:
        print("Invalid preset choice, using morning")
        params["preset"] = "morning"

    if mode_choice == "2":
        # Multiple runs mode
        try:
            num_runs_input = input("Enter number of runs (default: 10): ").strip()
            num_runs = int(num_runs_input) if num_runs_input else 10
        except (ValueError, EOFError):
            print("Using default: 10 runs")
            num_runs = 10

        try:
            steps_input = input("Enter steps per run (default: 100): ").strip()
            steps = int(steps_input) if steps_input else 100
        except (ValueError, EOFError):
            print("Using default: 100 steps")
            steps = 100

        # Temporarily disable animation for multiple runs
        original_animation = params["animation"]
        params["animation"] = False

        # Run multiple simulations
        run_multiple_simulations_and_plot(num_runs, steps)

        # Restore original animation setting
        params["animation"] = original_animation

    else:
        # Single simulation mode (default)
        run_simulation_and_send_json()

        if params["animation"]:
            model, anim = run_simulation_with_animation("H6_Traffic2.gif")