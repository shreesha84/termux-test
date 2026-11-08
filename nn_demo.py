import numpy as np
import time

# 1. Activation Function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Training Data (XOR Logic Gate)
# Input features: [Binary 1, Binary 2]
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Expected Outputs: 1 if inputs are different, 0 if they are the same
y = np.array([[0], [1], [1], [0]])

# 3. Initialize Network Architecture
np.random.seed(42) # For reproducible results
input_layer_neurons = 2
hidden_layer_neurons = 3
output_neurons = 1

# Initialize weights and biases with random numbers
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# 4. Hyperparameters
lr = 0.5        # Learning rate
epochs = 5000   # Number of training iterations

print("Training the neural network on your device...")
start_time = time.time()

# 5. Training Loop (Forward & Backward Propagation)
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_activations = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_activations, wout) + bout
    predicted_output = sigmoid(output_layer_input)
    
    # Backpropagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(wout.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_activations)
    
    # Updating Weights and Biases
    wout += hidden_layer_activations.T.dot(d_predicted_output) * lr
    bout += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden_layer) * lr
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    # Print loss tracking every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch:4d} | Mean Squared Error: {loss:.5f}")

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.4f} seconds!")

# 6. Final Test / Prediction Verification
print("\n--- Final Network Predictions ---")
for i in range(len(X)):
    print(f"Inputs: {X[i]} -> Target: {y[i][0]} -> Predicted: {predicted_output[i][0]:.4f}")
