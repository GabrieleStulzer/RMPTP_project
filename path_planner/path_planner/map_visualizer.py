#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray

import heapq

import numpy as np
from math import sin, cos, atan2, pi, sqrt, acos

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pprint import pprint

def mod2pi(theta):
    return theta % (2 * pi)

def sinc(x):
    if abs(x) < 0.002:
        return 1 - x**2 * (1/6 - x**2 / 120)
    else:
        return sin(x) / x

# Bipolar transform T and its inverse
def bipolar_transform(x, y, x0, y0, xf, yf):
    dx, dy = xf - x0, yf - y0
    phi = atan2(dy, dx)
    lambd = 0.5 * sqrt(dx**2 + dy**2)

    x_bar = x0 * cos(phi) + y0 * sin(phi) + lambd
    y_bar = -x0 * sin(phi) + y0 * cos(phi)

    R = np.array([[cos(phi), sin(phi)], [-sin(phi), cos(phi)]])
    p = np.array([x, y])
    c = np.array([x_bar, y_bar])

    return (1 / lambd) * (R @ (p - c)), phi, lambd

def inverse_bipolar_transform(x, y, phi, lambd, x0, y0):
    x_bar = x0 * cos(phi) + y0 * sin(phi) + lambd
    y_bar = -x0 * sin(phi) + y0 * cos(phi)

    R_inv = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
    p = lambd * np.array([x, y]) + np.array([x_bar, y_bar])

    return R_inv @ p

# Dubins path calculations (example: LSL)
def dubins_LSL(theta0, thetaf, kappa):
    C = cos(thetaf) - cos(theta0)
    S = 2 * kappa + sin(theta0) - sin(thetaf)
    s1 = mod2pi(atan2(C, S) - theta0) / kappa
    s3 = mod2pi(thetaf - atan2(C, S)) / kappa
    s2 = sqrt(2 + 4 * kappa**2 - 2 * cos(theta0 - thetaf) + 4 * kappa * (sin(theta0) - sin(thetaf))) / kappa
    return s1, s2, s3, s1 + s2 + s3

def dubins_RSR(theta0, thetaf, kappa):
    C = cos(theta0) - cos(thetaf)
    S = 2 * kappa - sin(theta0) + sin(thetaf)
    s1 = mod2pi(theta0 - atan2(C, S)) / kappa
    s3 = mod2pi(atan2(C, S) - thetaf) / kappa
    s2 = sqrt(2 + 4 * kappa**2 - 2 * cos(theta0 - thetaf) - 4 * kappa * (sin(theta0) - sin(thetaf))) / kappa
    return s1, s2, s3, s1 + s2 + s3

def dubins_LSR(theta0, thetaf, kappa):
    C = cos(theta0) + cos(thetaf)
    S = 2 * kappa + sin(theta0) + sin(thetaf)
    s2 = sqrt(-2 + 4 * kappa**2 + 2 * cos(theta0 - thetaf) + 4 * kappa * (sin(theta0) + sin(thetaf))) / kappa
    psi = atan2(-C, S) - atan2(-2, kappa * s2)
    s1 = mod2pi(psi - theta0) / kappa
    s3 = mod2pi(psi - thetaf) / kappa
    return s1, s2, s3, s1 + s2 + s3

def dubins_RSL(theta0, thetaf, kappa):
    C = cos(theta0) + cos(thetaf)
    S = 2 * kappa - sin(theta0) - sin(thetaf)
    s2 = sqrt(-2 + 4 * kappa**2 + 2 * cos(theta0 - thetaf) - 4 * kappa * (sin(theta0) + sin(thetaf))) / kappa
    psi = atan2(C, S) - atan2(2, kappa * s2)
    s1 = mod2pi(theta0 - psi) / kappa
    s3 = mod2pi(thetaf - psi) / kappa
    return s1, s2, s3, s1 + s2 + s3

def dubins_RLR(theta0, thetaf, kappa):
    C = cos(theta0) - cos(thetaf)
    S = 2 * kappa - sin(theta0) + sin(thetaf)
    arg = (1/8) * (6 - 4 * kappa**2 + 2 * cos(theta0 - thetaf) + 4 * kappa * (sin(theta0) - sin(thetaf)))
    if abs(arg) > 1: raise ValueError("RLR: invalid geometry")
    s2 = mod2pi(2 * pi - acos(arg)) / kappa
    s1 = mod2pi(theta0 - atan2(C, S) + 0.5 * kappa * s2) / kappa
    s3 = mod2pi(theta0 - thetaf + kappa * (s2 - s1)) / kappa
    return s1, s2, s3, s1 + s2 + s3

def dubins_LRL(theta0, thetaf, kappa):
    C = cos(thetaf) - cos(theta0)
    S = 2 * kappa + sin(theta0) - sin(thetaf)
    arg = (1/8) * (6 - 4 * kappa**2 + 2 * cos(theta0 - thetaf) - 4 * kappa * (sin(theta0) - sin(thetaf)))
    if abs(arg) > 1: raise ValueError("LRL: invalid geometry")
    s2 = mod2pi(2 * pi - acos(arg)) / kappa
    s1 = mod2pi(-theta0 + atan2(C, S) + 0.5 * kappa * s2) / kappa
    s3 = mod2pi(thetaf - theta0 + kappa * (s2 - s1)) / kappa
    return s1, s2, s3, s1 + s2 + s3

# All Dubins path generators
def all_dubins_paths(theta0, thetaf, kappa, lambd):
    paths = {}
    for name, fn in [
        ('LSL', dubins_LSL),
        ('RSR', dubins_RSR),
        ('LSR', dubins_LSR),
        ('RSL', dubins_RSL),
        ('RLR', dubins_RLR),
        ('LRL', dubins_LRL),
    ]:
        try:
            s1, s2, s3, L = fn(theta0, thetaf, kappa)
            paths[name] = np.array([s1, s2, s3, L]) * lambd
        except Exception as e:
            # Alcune geometrie non permettono una soluzione valida per RLR/LRL
            continue
    return paths


# Main function: from initial to final
def compute_dubins_path(checkpoint, checkpoint1, kappa_max):
    # Apply bipolar transform
    (x0, y0, theta0) = checkpoint
    (xf, yf, thetaf) = checkpoint1
    (p0, _), phi, lambd = bipolar_transform(x0, y0, x0, y0, xf, yf)
    (pf, _), _, _ = bipolar_transform(xf, yf, x0, y0, xf, yf)

    # Transform angles and curvature
    theta0_std = theta0 - phi
    thetaf_std = thetaf - phi
    kappa_std = lambd * kappa_max

    # Solve standard Dubins problem
    paths = all_dubins_paths(mod2pi(theta0_std), mod2pi(thetaf_std), kappa_std, lambd)
    optimal = min(paths.items(), key=lambda x: x[1][3])
    return optimal, phi, lambd


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
            '/shelfino3/amcl_pose',
            self.update_robot_pose,
            qos_profile
        )

        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/shelfino1/amcl_pose',
            self.update_robot_pose_1,
            qos_profile
        )

        self.goal_sub = self.create_subscription(
            PoseArray,
            '/gates',
            self.update_goal_pose,
            qos_profile
        )

        self.resolution = 0
        self.cell_size = 0

        self.grid_data = None
        self.grid_info = None
        self.goal_pose = None
        self.robot_pose = None
        self.robot_pose_1 = None
        self.decomposed_grid = None

    def update_occupancy_grid(self, msg):
        self.grid_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        # self.grid_data = np.flipud(self.grid_data)
        self.grid_info = msg.info
        self.resolution = self.grid_info.resolution
        self.plot()
    
    def update_robot_pose(self, msg):
        print("Received robot Pose")
        print(msg)
        rot = msg.pose.pose.orientation
        theta = atan2(2 * (rot.w * rot.z + rot.x * rot.y), 1 - 2 * (rot.y**2 + rot.z**2))
        self.robot_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, theta)
        self.plot()
    
    def update_goal_pose(self, msg):
        print("Received Goal Pose")
        self.goal_pose = (msg.poses[0].position.x, msg.poses[0].position.y)
        self.plot()

    def update_robot_pose_1(self, msg):
        print("Received robot1 Pose")
        rot = msg.pose.pose.orientation
        theta = atan2(2 * (rot.w * rot.z + rot.x * rot.y), 1 - 2 * (rot.y**2 + rot.z**2))
        self.robot_pose_1 = (msg.pose.pose.position.x, msg.pose.pose.position.y, theta)
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
            # return abs(a[0] - b[0]) + abs(a[1] - b[1])
            return np.hypot((a[0] - b[0]), (a[1] - b[1]))

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

            print(neighbors)

            for neighbor in neighbors:
                # if 0 <= neighbor[0] < decomposed_grid.shape[0] and 0 <= neighbor[1] < decomposed_grid.shape[1]:
                # print(f"{decomposed_grid[neighbor[0], neighbor[1]]=}, {neighbor[0]}, {neighbor[1]}")
                if decomposed_grid[neighbor[1] + 23, neighbor[0] + 25] == 0:
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        return []  # No path found
    
    
    def grid2pos(self, indexes, resolution, cell_size):
        x = (indexes[0] + 0.5) * resolution * cell_size
        y = (indexes[1] + 0.5) * resolution * cell_size
        return x, y
        
    
    def pos2grid(self, position, resolution, cell_size):
        x = int(position[0] / (cell_size * resolution)) - (1 if (position[0] < 0 and position[0] % (cell_size * resolution) > 0) else 0)
        y = int(position[1] / (cell_size * resolution)) - (1 if (position[1] < 0 and position[1] % (cell_size * resolution) > 0) else 0)
        return (x, y)
    
    
    def plot_grid(self, origin, cell_size, resolution):
        for i in range(self.decomposed_grid.shape[0]):
            for j in range(self.decomposed_grid.shape[1]):
                if self.decomposed_grid[i, j] > 0:
                    x = origin.x + j * cell_size * resolution 
                    y = origin.y + i * cell_size * resolution

                    plt.gca().add_patch(
                        plt.Rectangle(
                            (x, y),
                            cell_size * resolution,
                            cell_size * resolution,
                            fill=False,  # Only draw the border
                            edgecolor="green",
                            linewidth=1
                        )
                    )
    
    
    def plot_path(self, path, cell_size, resolution):
        for position in path[1:-1]:
            plt.gca().add_patch(
                plt.Rectangle(
                    (position[0] * cell_size * resolution, position[1] * cell_size * resolution),
                    cell_size * resolution,
                    cell_size * resolution,
                    facecolor="red",  # Only draw the border
                    edgecolor="green",
                    linewidth=1
                )
        )
        plt.gca().add_patch(
            plt.Rectangle(
                (path[0][0] * cell_size * resolution, path[0][1] * cell_size * resolution),
                cell_size * resolution,
                cell_size * resolution,
                facecolor="orange",  # Only draw the border
                edgecolor="green",
                linewidth=1
            )
        )
        plt.gca().add_patch(
            plt.Rectangle(
                (path[-1][0] * cell_size * resolution, path[-1][1] * cell_size * resolution),
                cell_size * resolution,
                cell_size * resolution,
                facecolor="blue",  # Only draw the border
                edgecolor="green",
                linewidth=1
            )
        )
        plt.gca().add_patch(
            plt.Rectangle(
                (0, 0),
                cell_size * resolution,
                cell_size * resolution,
                facecolor="blue",  # Only draw the border
                edgecolor="green",
                linewidth=1
            )
        )
        
    
    def plotLdubin(self, x0, y0, radius, theta0, arc_len):
        center = (x0 - radius * sin(theta0), y0 + radius * cos(theta0))          # Circle center at (x, y)
        theta1 = atan2(-cos(theta0), sin(theta0))               # Start angle in degrees
        theta2 = theta1 + arc_len / radius              # End angle in degrees
        n_arc_points = 100         # Number of points used to draw the arc
        
        angles = np.linspace(theta1, theta2, n_arc_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        plt.plot(x, y, color='orange', linewidth=2, label='Arc')
        xf = center[0] + radius * cos(theta2)
        yf = center[1] + radius * sin(theta2)
        return xf, yf, theta0 + theta2 - theta1
        
    def plotRdubin(self, x0, y0, radius, theta0, arc_len):
        center = (x0 + radius * sin(theta0), y0 - radius * cos(theta0))          # Circle center at (x, y)
        theta1 = atan2(cos(theta0), -sin(theta0))               # Start angle in degrees
        theta2 = theta1 - arc_len / radius              # End angle in degrees
        n_arc_points = 100         # Number of points used to draw the arc
        
        angles = np.linspace(theta1, theta2, n_arc_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        plt.plot(x, y, color='orange', linewidth=2, label='Arc')
        xf = center[0] + radius * cos(theta2)
        yf = center[1] + radius * sin(theta2)
        return xf, yf, theta0 + theta2 - theta1
    
    def plotSdubin(self, x0, y0, theta0, s_len):
        n_line_points = 100         # Number of points used to draw the line
        distances = np.linspace(0, s_len, n_line_points)
        x = x0 + np.cos(theta0) * distances
        y = y0 + np.sin(theta0) * distances
        plt.plot(x, y, color='orange', linewidth=2, label='Line')
        xf = x0 + s_len * cos(theta0)
        yf = y0 + s_len * sin(theta0)
        return xf, yf
    
    def plot_dubin(self, x0, y0, radius, theta0, name, lengths):
        x = x0
        y = y0
        theta = theta0
        for i, letter in enumerate(name):
            if letter == "L":
                x, y, theta = self.plotLdubin(x, y, radius, theta, lengths[i])
            elif letter == "R":
                x, y, theta = self.plotRdubin(x, y, radius, theta, lengths[i])
            elif letter == "S":
                x, y = self.plotSdubin(x, y, theta, lengths[i])
        return x, y, theta
    
    def shortest_path_robot(self, robot_pose, goal_pose):
        # find the closest cell to start point
        start_index = self.pos2grid(robot_pose, self.resolution, self.cell_size)
        goal_index = self.pos2grid(goal_pose, self.resolution, self.cell_size)
        print(start_index)
        print(goal_index)

        path = self.a_star(self.decomposed_grid, start_index, goal_index)
        if path:
            print(path)
            self.plot_path(path, self.cell_size, self.resolution)

        return path
        

    def path_to_dubins(self, path, start, goal):
        checkpoints = []
        checkpoints.append(start)
        
        for i in range(len(path) - 2):
            current = np.array(self.grid2pos(path[i], self.resolution, self.cell_size))
            next = np.array(self.grid2pos(path[i + 1], self.resolution, self.cell_size))
            # print(current, next, self.grid2pos(current, resolution, cell_size), self.grid2pos(next, resolution, cell_size))

            diff = next - current
            angle = atan2(diff[1], diff[0])
            mid = (current + next) / 2

            checkpoints.append((mid[0], mid[1], angle))
        checkpoints.append(goal)

        # plot a dubins curve
        # x0, y0, theta0 = 0, 0, -pi/2
        # xf, yf, thetaf = 4, 0, -pi/2
        for i in range(len(checkpoints) - 1):
            x0, y0, theta0 = checkpoints[i]
            xf, yf, thetaf = checkpoints[i + 1]
            min_radius = self.cell_size * self.resolution / 2 / 1.1
        
            kappa_max = 1.0 / min_radius
            result, phi, lambd = compute_dubins_path(x0, y0, theta0, xf, yf, thetaf, kappa_max)
            print(self.plot_dubin(x0, y0, min_radius, theta0, result[0], result[1]))

    def plot(self):
        if self.grid_data is None or self.grid_info is None:
            return
        if self.robot_pose is None and self.robot_pose_1 is None and self.goal_pose is None:
            print("Missing infos to proceed")
            return

        # Extract metadata
        width = self.grid_info.width
        height = self.grid_info.height
        resolution = self.grid_info.resolution
        origin = self.grid_info.origin.position

        # Convert data to binary occupancy grid: 0 = free (white), >0 = occupied (black)
        binary_grid = np.where(self.grid_data > 0, 0, 1)

        # Apply fixed cell decomposition
        self.cell_size = 100  # Example cell size
        if self.decomposed_grid is None:
            self.decomposed_grid = self.apply_fixed_cell_decomposition(binary_grid, self.cell_size)


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
        
        # Display the decomposed grid
        self.plot_grid(origin, self.cell_size, self.resolution)

        # if p:=self.robot_pose:
        #     start = p
        # else:
        #     return
        
        # start = (6.1, -4.3, 0.0)
        # goal = (-3, -6, 0)  # Example goal

        # # find the closest cell to start point
        # start_index = self.pos2grid(start, resolution, self.cell_size)
        # goal_index = self.pos2grid(goal, resolution, self.cell_size)

        # path = self.a_star(self.decomposed_grid, start_index, goal_index)
        # print(start, goal, path)
        # self.plot_path(path, self.cell_size, resolution)
        
        # print("checkpoints")
        # checkpoints = []
        # checkpoints.append(start)
        
        # for i in range(len(path) - 2):
        #     current = np.array(self.grid2pos(path[i], resolution, self.cell_size))
        #     next = np.array(self.grid2pos(path[i + 1], resolution, self.cell_size))
        #     # print(current, next, self.grid2pos(current, resolution, cell_size), self.grid2pos(next, resolution, cell_size))

        #     diff = next - current
        #     angle = atan2(diff[1], diff[0])
        #     mid = (current + next) / 2

        #     checkpoints.append((mid[0], mid[1], angle))
        # checkpoints.append(goal)

        
        # for checkpoint in checkpoints:
        #     print(checkpoint)
        #     plt.plot(
        #         checkpoint[0],
        #         checkpoint[1],
        #         'bo',  # Red dot for the robot position
        #         label="Robot Position"
        #     )

        # print("end checkpoints")
        
        print(self.robot_pose)
        print(self.robot_pose_1)
        print(self.robot_pose_1)
        
        # Plot robot position if available
        if self.robot_pose and self.robot_pose_1 and self.goal_pose:
            print("Plotting Robot Pose")
            plt.plot(
                self.robot_pose[0],
                self.robot_pose[1],
                'ro',  # Red dot for the robot position
                label="Robot Position"
            )

            print("Computing Robot Path")
            robot1_path = self.shortest_path_robot(self.robot_pose, self.goal_pose)
            print(robot1_path)
            if robot1_path:
                self.path_to_dubins(robot1_path, self.robot_pose, self.goal_pose)

            plt.plot(
                self.robot_pose_1[0],
                self.robot_pose_1[1],
                'bo',  # Red dot for the robot position
                label="Robot Position"
            )
            # robot2_path = self.shortest_path_robot(self.robot_pose_1, self.goal_pose)
            # self.path_to_dubins(robot2_path, self.robot_pose_1, self.goal_pose)

            plt.plot(
                self.goal_pose[0],
                self.goal_pose[1],
                'go',  # Green dot for the goal position
                label="Goal Position"
            )
            plt.legend()


            plt.title("Decomposed Grid with Robot Position")
            plt.xlabel("Y (meters)")
            plt.ylabel("X (meters)")
            plt.xlim([origin.x, origin.x + width * resolution])
            plt.ylim([origin.y, origin.y + height * resolution])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
            exit()
        
        else:
            print("Missing info to proceed")
            return


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
