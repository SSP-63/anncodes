import numpy as np

# Initialize parameters
c = 1  # Learning rate
W = np.array([0, 0, 0])  # Initial weights (including bias weight)
epochs = 10  # Number of iterations

# Define inputs and expected outputs (truth table for AND gate)
inputs = np.array([
    [-1, -1, 1],  # x1 = -1, x2 = -1, bias = 1
    [-1,  1, 1],  # x1 = -1, x2 =  1, bias = 1
    [ 1, -1, 1],  # x1 =  1, x2 = -1, bias = 1
    [ 1,  1, 1]   # x1 =  1, x2 =  1, bias = 1
])

outputs = np.array([-1, -1, -1, 1])  # Expected outputs for AND gate

# Perceptron Learning Rule
for epoch in range(epochs):
    errors = 0
    for i in range(len(inputs)):
        x = inputs[i]
        y = outputs[i]
        net = np.dot(W, x)
        y_pred = 1 if net > 0 else -1  # Activation function (sign function)
        
        if y_pred != y:  # If prediction is wrong, update weights
            W = W + c * (y - y_pred) * x
            errors += 1

    print(f"Epoch {epoch+1}: Weights = {W}, Errors = {errors}")
    if errors == 0:
        break  # Stop if no errors (convergence)

# Testing the trained model
def predict(x):
    net = np.dot(W, x)
    return 1 if net > 0 else -1  # Activation function

# Test the AND gate
print("\nTesting the trained Perceptron for AND gate:")
for i in range(len(inputs)):
    x = inputs[i]
    y_pred = predict(x)
    print(f"Input: {x[:2]} -> Predicted Output: {y_pred}, Expected Output: {outputs[i]}")
