# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.path import Path

from math import pi, sin, cos
def deg2rad(deg):
    return (deg/180.0)*pi

def get_scale_matrix(sx, sy):
    return np.array([[sx, 0.0],
                     [0.0, sy]])
def get_rotation_matrix(deg):
    rad = deg2rad(deg)
    return np.array([   [cos(rad), -sin(rad)],
                        [sin(rad), cos(rad)]])
    
# Define the map size
map_size = (120, 100)

# Define start and goal points
start = (70, 20)
goal = (115, 70)
goal_radius = 10  # How close we need to be to the goal to stop
phi = 90

# RRT parameters
step_size = 12
r = 6
max_iterations = 5000
new_alg = True

import math

obstacles = [ # Get from local frames (polygons)
    # [[20, 60], [20, 80], [30, 80], [40, 70], [40, 60]],
    # [[30, 30], [30, 50], [50, 50], [50, 30]],
    # [[60, 50], [50, 60], [60, 70], [70, 60]],
    # [[70, 10], [60, 20], [70, 30], [80, 20]],
    [[0, 0], [40, 0], [42, 10], [38, 20], [35, 30], [30, 33], [22, 37], [18,45], [20, 60], [22, 75], [12, 80], [0, 80]], #1
    [[0, 82], [12, 84], [16, 92], [14, 100], [0, 100]], #2
    [[30, 100], [35, 80], [40, 82], [45, 80], [52, 80], [60, 85], [65,82], [75, 85], [85,80], [88, 82], [92,78], [100, 75], [108, 80], [120, 80], [120, 100] ], #3
    [[70,35], [72, 45], [68,55], [70, 60], [75, 62], [80, 57], [85, 54], [90,40], [88,35], [80, 30], [75, 30]  ], #4
    [[120, 60], [105, 60], [100, 55], [95, 50], [97, 47], [94, 44], [96, 40], [92, 38], [94,32], [100, 30], [110, 32], [120, 30] ], #5
    [[60, 0], [60, 5], [62, 10], [70, 12], [80, 10], [90, 12], [100, 10], [110, 10], [120, 12], [120, 0]] #6
]

# Visualization
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, map_size[0])
ax.set_ylim(0, map_size[1])

# Plot obstacles
for polygon in obstacles:
    # Extract polygon coordinates
    polygon_x, polygon_y = zip(*polygon)

    # Close the polygon by adding the first point at the end
    polygon_x += (polygon_x[0],)
    polygon_y += (polygon_y[0],)
    ax.fill(polygon_x, polygon_y, color='red', alpha=0.6)

plt.legend()
# plt.title("RRT Path Planning in 2D")
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])


# Helper function to check collision with obstacles
def is_collision_free(point, margin):
    x, y = point
    if x < 0 or x >= map_size[0]:
        return False
    if y < 0 or y >= map_size[1]:
        return False
    for obstacle in obstacles:
        # n = len(obstacle)
        obstacle = np.asarray(obstacle) # n x 2
        # print(obstacle)
        center = np.mean(obstacle, axis=0)
        origin_points = obstacle - center # at orgin # nx2
        lengths = np.linalg.norm(origin_points, axis=1)
        avg_lengths = np.mean(lengths)
        # scale_factor = (avg_lengths + step_size/2.0)/avg_lengths
        scale_factor = 1.0
        origin_points2 = origin_points@get_scale_matrix(scale_factor, scale_factor)
        obstacle2 = origin_points2 + center # move back to old center
        # print(obstacle2)
        poly_path = Path(obstacle2)  # Create a path from polygon vertices

        
        xc, yc = point
        for i in range(15):
            x = xc + np.random.uniform(low=-r, high=r)
            y1 = yc +  math.sqrt(r**2 - (x-xc)**2)    
            y2 = yc + -math.sqrt(r**2 - (x-xc)**2)    
            if poly_path.contains_point([x, y1]):
                return False
            if poly_path.contains_point([x, y2]):
                return False
    return True

# RRT algorithm
class RRT:
    def __init__(self, start, goal, step_size, max_iterations):
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.nodes = [start]  # List of explored nodes
        self.parent = {start: None}  # Parent dictionary for backtracking

    def get_random_point_perspective(self):
        from_node = self.nodes[-1]
        vec = np.array(self.goal) - np.array(from_node)
        deg = random.randint(-phi, phi)
        steps = random.randint(1, 5)
        vec_rot = get_rotation_matrix(deg)@vec.reshape((2,1)) # (2,1)
        vec_rot = vec_rot.reshape((2,))
        length = np.linalg.norm(vec_rot)
        direction = vec_rot / length
        return np.array(from_node) + (self.step_size*steps) * direction
        
    
    def get_random_point(self):
        x = random.randint(0, map_size[0])
        y = random.randint(0, map_size[1])
        return np.array([x, y]).reshape((2,))

    def get_nearest_node(self, point):
        """Find the nearest node in the tree."""
        return min(self.nodes, key=lambda n: np.linalg.norm(np.array(n) - np.array(point)))

    def steer(self, from_node, to_node=None):
        vec = np.array(to_node) - np.array(from_node)
        length = np.linalg.norm(vec)
        if length == 0:
            return from_node
        direction = vec / length
        new_node = np.array(from_node) + self.step_size * direction
        return tuple(new_node.astype(int))

    def run(self):
        """Run the RRT algorithm to find a path."""
        for i in range(self.max_iterations):
            if new_alg:
                rand_point = self.get_random_point_perspective()
            else:
                rand_point = self.get_random_point()
            nearest_node = self.get_nearest_node(rand_point) # nearest from tree to rand_point
            new_node = self.steer(nearest_node, rand_point)

            if is_collision_free(new_node, margin=step_size/2.0):
                self.nodes.append(new_node)
                self.parent[new_node] = nearest_node
                # print(f'i = {i}, new_node = {new_node}')
                # Check if we reached the goal region
                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) < goal_radius:
                    self.parent[self.goal] = new_node
                    print(f'Stop at iteration = {i}')
                    return self.get_path()
            # print(f'i = {i}, new_node = {new_node}')
        return None  # No path found

    def get_path(self):
        """Backtrack from goal to start to extract the path."""
        path = []
        node = self.goal
        while node is not None:
            path.append(node)
            node = self.parent[node]
        return path[::-1]  # Reverse to get start-to-goal path

# Run RRT
rrt = RRT(start, goal, step_size, max_iterations)

import time
t1 = time.time()
path = rrt.run()
t2 = time.time()
print(f'Runtime: {(t2-t1)*1000: 0.2f}')

# Plot RRT tree
for node, parent in rrt.parent.items():
    if parent:
        plt.plot([node[0], parent[0]], [node[1], parent[1]], 'g-', alpha=0.5)

# Plot path
if path:
    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, 'b-', linewidth=2, label="Path")
    x = np.asanyarray(path_x)
    y = np.asanyarray(path_y)
    xy = np.stack([x, y], axis=0).T # nx2
    n = len(xy)
    xy_s = xy[0]
    total = 0.0
    for i in range(1, n):
        xy_e = xy[i]
        vec = xy_e - xy_s
        dist = np.sqrt(np.dot(vec, vec))
        total += dist
    print(f'travel distance = {total}')
    # print(xy)
    # dist = np.sqrt()

# Plot start and goal
plt.scatter(*start, color="blue", s=100, label="Start")
plt.scatter(*goal, color="green", s=100, label="Goal", marker="X")

plt.legend()
plt.title("RRT Path Planning in 2D")
# plt.grid()
plt.show()
