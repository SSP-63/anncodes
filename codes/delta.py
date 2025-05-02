import numpy as np

def delta_learning(inputs, desired_outputs, initial_weights, learning_rate=0.1, lambda_val=1):
    weights = initial_weights.copy()
    
    for i, (x, d) in enumerate(zip(inputs, desired_outputs)):
        print(f"\nStep {i+1}: Processing input {x} with desired output {d}")
        
        net = np.dot(weights, x)
        net = round(net, 3)
        print(f"  Net = w^T * x = {net}")
        
        fnet = round((2 / (1 + np.exp(-lambda_val * net))) - 1, 3)  # Bipolar sigmoid activation
        print(f"  f(Net) = {fnet}")
        
        fnet_derivative = round(0.5 * (1 - fnet ** 2), 3)  # Derivative of activation function
        print(f"  f'(Net) = {fnet_derivative}")
        
        weight_change = learning_rate * (d - fnet) * fnet_derivative * x
        weight_change = np.round(weight_change, 3)  # Round each element
        weights += weight_change
        
        print(f"  Updated Weights w{i+2} = {np.round(weights, 3)}")

# Given initial weight vector and input vectors
initial_weights = np.array([1, -1, 0, 0.5])
inputs = np.array([
    [1, -2, 0, -1],
    [0, 1.5, -0.5, -1],
    [-1, 1, 0.5, -1]
])
desired_outputs = np.array([-1, -1, 1])  # Given desired outputs

delta_learning(inputs, desired_outputs, initial_weights, learning_rate=0.1, lambda_val=1)
