import numpy as np

def widrow_hoff_learning(inputs, desired_outputs, initial_weights, learning_rate=0.1):
    weights = initial_weights.copy()
    
    for i, (x, d) in enumerate(zip(inputs, desired_outputs)):
        print(f"\nStep {i+1}: Processing input {x} with desired output {d}")
        
        net = np.dot(weights, x)
        print(f"  Net = w^T * x = {np.round(net,3)}")
        
        r = d - net  # Compute error (residual)
        print(f"  Residual (r) = d - Net = {np.round(r,3)}")
        
        weights += learning_rate * r * x  # Update weights using delta rule
        print(f"  Updated Weights w{i+2} = {np.round(weights,3)}")

# Given initial weight vector and input vectors
initial_weights = np.array([1, -1, 0, 0.5])
inputs = np.array([
    [1, -2, 1.5, 0],
    [1, -0.5, -2, -1.5],
    [0, 1, -1, 1.5]
])

# Desired outputs for each input vector
desired_outputs = np.array([0.5, -0.3, 0.8])  # Adjust based on problem requirements

# Set learning rate
learning_rate = 0.1  

widrow_hoff_learning(inputs, desired_outputs, initial_weights, learning_rate)
