import numpy as np

def perceptron_learning(inputs, desired_outputs, initial_weights, learning_rate=1.0):
    weights = initial_weights.copy()
    
    for i, (x, d) in enumerate(zip(inputs, desired_outputs)):
        print(f"\nStep {i+1}: Processing input {x} with desired output {d}")
        
        net = np.dot(weights, x)
        print(f"  Net = w^T * x = {net}")
        
        fnet = np.sign(net)
        print(f"  f(Net) = sign(Net) = {fnet}")
        
        weight_change = learning_rate * (d - fnet) * x
        weights += weight_change
        
        if np.all(weight_change == 0):
            print(f"  No weight update (f(Net) matches desired output)")
        else:
            print(f"  Updated Weights w{i+2} = {weights}")

# Given initial weight vector and input vectors
initial_weights = np.array([1, -1, 0, 0.5])
inputs = np.array([
    [1, -2, 0, -1],
    [0, 1.5, -0.5, -1],
    [-1, 1, 0.5, -1]
])
desired_outputs = np.array([-1, -1, 1])  # Given desired outputs

perceptron_learning(inputs, desired_outputs, initial_weights, learning_rate=0.1)
