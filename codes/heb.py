import numpy as np

def hebbian_learning_verbose(inputs, initial_weights):
    weights = initial_weights.copy()
    
    for i, x in enumerate(inputs):
        print(f"\nStep {i+1}: Processing input {x}")
        
        net = np.dot(weights, x)
        print(f"  Net = w^T * x = {net}")
        
        fnet = np.sign(net)
        print(f"  f(Net) = sign(Net) = {fnet}")
        
        weights += fnet * x
        print(f"  Updated Weights w{i+2} = {weights}")

# Given initial weight vector and input vectors
initial_weights = np.array([1, -1, 0, 0.5])
inputs = np.array([
    [1, -2, 1.5, 0],
    [1, -0.5, -2, -1.5],
    [0, 1, -1, 1.5]
])

hebbian_learning_verbose(inputs, initial_weights)
