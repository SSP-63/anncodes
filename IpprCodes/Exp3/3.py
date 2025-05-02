import numpy as np

# Given point
P = np.array([3, 4])

# Scaling factors (Sx, Sy)
S = np.array([2, 3])

# Apply scaling
P_new = P * S

print("New coordinates after scaling:", P_new)
