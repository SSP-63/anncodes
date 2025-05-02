import numpy as np

# Given point
P = np.array([2, 3])

# Shear factor along x-axis
shear_factor = 2

# Shearing transformation matrix
shear_matrix = np.array([[1, shear_factor],
                          [0, 1]])

# Apply shearing
P_new = np.dot(shear_matrix, P)

print("New coordinates after shearing:", P_new)
