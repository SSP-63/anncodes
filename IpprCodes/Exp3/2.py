import numpy as np

# Given point
P = np.array([2, 3])

# 90-degree counterclockwise rotation matrix
theta = np.radians(90)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

# Apply rotation
P_new = np.dot(rotation_matrix, P)

print("New coordinates after rotation:", P_new)
