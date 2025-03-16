#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped

import heapq

import numpy as np
import matplotlib.pyplot as plt

class OccupancyGridVisualizer(Node):
    def __init__(self):
        super().__init__('occupancy_grid_visualizer')
        self.grid_subscription = self.create_subscription(
            OccupancyGrid,
            '/shelfino1/global_costmap/costmap',
            self.update_occupancy_grid,
            10
        )

        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/shelfino1/amcl_pose',
            self.update_robot_pose,
            qos_profile
        )
        self.grid_data = None
        self.grid_info = None
        self.robot_pose = None
        self.decomposed_grid = None

    def update_occupancy_grid(self, msg):
        self.grid_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        # self.grid_data = np.flipud(self.grid_data)
        self.grid_info = msg.info
        self.plot()

    def update_robot_pose(self, msg):
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.plot()

    def apply_fixed_cell_decomposition(self, grid, cell_size):
        """Apply fixed cell decomposition to the grid."""
        height, width = grid.shape
        cell_rows = height // cell_size
        cell_cols = width // cell_size

        # Aggregate cells into a decomposed grid
        decomposed_grid = np.zeros((cell_rows, cell_cols), dtype=int)
        for i in range(cell_rows):
            for j in range(cell_cols):
                cell = grid[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                decomposed_grid[i, j] = 0 if np.any(cell == 0) else 1

        return decomposed_grid
    
    def a_star(self, decomposed_grid, start, goal):
        """A* algorithm to find the shortest path on the decomposed grid."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            neighbors = [
                (current[0] + 1, current[1]),
                (current[0] - 1, current[1]),
                (current[0], current[1] + 1),
                (current[0], current[1] - 1)
            ]

            for neighbor in neighbors:
                if 0 <= neighbor[0] < decomposed_grid.shape[0] and 0 <= neighbor[1] < decomposed_grid.shape[1]:
                    if decomposed_grid[neighbor[0], neighbor[1]] == 1:
                        continue

                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def plot(self):
        if self.grid_data is None or self.grid_info is None:
            return

        # Extract metadata
        width = self.grid_info.width
        height = self.grid_info.height
        resolution = self.grid_info.resolution
        origin = self.grid_info.origin.position

        # Convert data to binary occupancy grid: 0 = free (white), >0 = occupied (black)
        binary_grid = np.where(self.grid_data > 0, 0, 1)

        # Apply fixed cell decomposition
        cell_size = 100  # Example cell size
        if self.decomposed_grid is None:
            self.decomposed_grid = self.apply_fixed_cell_decomposition(binary_grid, cell_size)

        # Display the decomposed grid
        plt.figure(figsize=(10, 10))
        plt.imshow(
            binary_grid,
            cmap="gray",
            origin="upper",
            extent=[
                origin.x,
                origin.x + width * resolution,  # Adjust for new proportions
                origin.y + height * resolution,
                origin.y,
            ],
        )
        for i in range(self.decomposed_grid.shape[0]):
            for j in range(self.decomposed_grid.shape[1]):
                if self.decomposed_grid[i, j] > 0:
                    x = origin.x + j * cell_size * resolution 
                    y = -origin.y - (self.decomposed_grid.shape[0] - i - 1) * cell_size * resolution

                    plt.gca().add_patch(
                        plt.Rectangle(
                            (x, y),
                            cell_size * resolution,
                            -cell_size * resolution,
                            fill=False,  # Only draw the border
                            edgecolor="green",
                            linewidth=1
                        )
                    )

        start = (2,2)
        goal = (4, 4)  # Example goal
        path = self.a_star(self.decomposed_grid, start, goal)

        # Draw cell start and goal with an organge color
        start_x = start[0] * cell_size * resolution
        start_y = (self.decomposed_grid.shape[0] - start[1]) * cell_size * resolution
        # plt.plot(start[0], start[1], 'go', label="Start Position")
        goal_x = goal[0] * cell_size * resolution
        goal_y = (self.decomposed_grid.shape[0] - goal[1]) * cell_size * resolution
        plt.plot(goal[0], goal[1], 'ro', label="Goal Position")

        # finde closest cell to start point
        start = (int(start[0] / (cell_size * resolution)), int(start[1] / (cell_size * resolution)))  # Example start
        plt.gca().add_patch(
            plt.Rectangle(
                (start[0], start[1]),
                cell_size * resolution,
                -cell_size * resolution,
                fill="red",  # Only draw the border
                edgecolor="green",
                linewidth=1
            )
        )
        goal = (int(goal[0] / (cell_size * resolution)), int(goal[1] / (cell_size * resolution)))  # Example start
        plt.gca().add_patch(
            plt.Rectangle(
                (goal[0], goal[1]),
                cell_size * resolution,
                -cell_size * resolution,
                fill="blue",  # Only draw the border
                edgecolor="green",
                linewidth=1
            )
        )

        # Plot robot position if available
        if self.robot_pose:
            plt.plot(
                self.robot_pose[0],
                self.robot_pose[1],
                'ro',  # Red dot for the robot position
                label="Robot Position"
            )
            plt.plot(
                -5,
                -5,
                'go',  # Green dot for the goal position
                label="Goal Position"
            )
            plt.legend()


            # Plan a path using A*
            # start = (int((self.robot_pose[0]-origin.x) / (cell_size * resolution)), int((self.robot_pose[1] - origin.y) / (cell_size * resolution)))  # Example start
            # start = (int((3-origin.x) / (cell_size * resolution)), int((3 - origin.y) / (cell_size * resolution)))  # Example start
            # end = (int((-5-origin.x) / (cell_size * resolution)), int((-5 - origin.y) / (cell_size * resolution)))  # Example start
            start = (2,2)
            goal = (4, 4)  # Example goal
            path = self.a_star(self.decomposed_grid, start, goal)

            # Draw cell start and goal with an organge color
            # start_x = origin.x + start[0] * cell_size * resolution
            # start_y = -origin.y - (self.decomposed_grid.shape[0] - start[1]) * cell_size * resolution
            # plt.plot(start_x, start_y, 'go', label="Start Position")
            # goal_x = origin.x + goal[0] * cell_size * resolution
            # goal_y = -origin.y - (self.decomposed_grid.shape[0] - goal[1]) * cell_size * resolution
            # plt.plot(goal_x, goal_y, 'ro', label="Goal Position")



            if path:
                print("Path found:", path)
                path_x = [origin.x + p[1] * cell_size * resolution for p in path]
                path_y = [-origin.y - (self.decomposed_grid.shape[0] - p[0]) * cell_size * resolution for p in path]
                plt.plot(path_x, path_y, 'b-', label="Planned Path")
                plt.legend()
        


        plt.title("Decomposed Grid with Robot Position")
        plt.xlabel("Y (meters)")
        plt.ylabel("X (meters)")
        plt.xlim([origin.x, origin.x + width * resolution])
        plt.ylim([origin.y, origin.y + height * resolution])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
