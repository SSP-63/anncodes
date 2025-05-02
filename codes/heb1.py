import numpy as np

def hebbian_learning_bipolar(inputs, initial_weights, lambda_value=1, learning_rate=0.1):
    weights = initial_weights.copy()
    
    for i, x in enumerate(inputs):
        print(f"\nStep {i+1}: Processing input {x}")
        
        net = np.dot(weights, x)
        print(f"  Net = w^T * x = {net}")
        
        fnet = (2 / (1 + np.exp(-lambda_value * net))) - 1
        print(f"  f(Net) = (2 / (1 + e^(-lambda * Net))) - 1 = {fnet}")
        
        weights += learning_rate * fnet * x  # Updated weight rule with learning rate
        print(f"  Updated Weights w{i+2} = {np.round(weights, 3)}")

# Given initial weight vector and input vectors
initial_weights = np.array([1, -1, 0, 0.5])
inputs = np.array([
    [1, -2, 1.5, 0],
    [1, -0.5, -2, -1.5],
    [0, 1, -1, 1.5]
])

# Set lambda value (activation slope) and learning rate
lambda_value = 1    # Adjust to control activation steepness
learning_rate = 1  # Controls update step size

hebbian_learning_bipolar(inputs, initial_weights, lambda_value, learning_rate)
