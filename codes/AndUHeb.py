import numpy as np

# Initialize parameters
c = 1  # Learning rate
W = np.array([0, 0, 0])  # Initial weights (including bias weight)

# Define inputs and expected outputs (truth table for AND gate)
inputs = np.array([
    [-1, -1, 1],  # x1 = -1, x2 = -1, bias = 1
    [-1,  1, 1],  # x1 = -1, x2 =  1, bias = 1
    [ 1, -1, 1],  # x1 =  1, x2 = -1, bias = 1
    [ 1,  1, 1]   # x1 =  1, x2 =  1, bias = 1
])

outputs = np.array([-1, -1, -1, 1])  # Expected outputs for AND gate

# Hebbian Learning Rule: W_new = W_old + c * y * x
for i in range(len(inputs)):
    x = inputs[i]
    y = outputs[i]
    W = W + c * y * x
    print(f"Updated weights after sample {i+1}: {W}")

# Testing the trained model
def predict(x):
    net = np.dot(W, x)
    return 1 if net > 0 else -1  # Activation function (sign function)

# Test the AND gate
print("\nTesting the trained AND gate:")
for i in range(len(inputs)):
    x = inputs[i]
    y_pred = predict(x)
    print(f"Input: {x[:2]} -> Predicted Output: {y_pred}, Expected Output: {outputs[i]}")
