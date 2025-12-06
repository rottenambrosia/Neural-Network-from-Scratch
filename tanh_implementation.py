import utils
from utils import *
import nnfs
from nnfs import datasets
nnfs.init()

X, y = nnfs.datasets.spiral_data(1000, 3)
# Build model
dense1 = Dense(2, 64, "tanh")
dense2 = Dense(64, 64, "tanh")
dense3 = Dense(64, 64, "tanh")
output = Dense(64, 3, "softmax")  # Changed to 3 output units for 3 classes

# Activation functions
act_tanh = tanh()
act_softmax = Softmax()

# Loss
loss = SoftmaxCategoricalCrossentropy()
all_losses = []
all_accuracy = []
# Optimizer
optimizer = GradientDescentOptimizer(learning_rate=0.1, decay=5e-4)

# Training loop
for epoch in range(1000):
    # Forward pass
    z1 = dense1.forward(X)
    a1 = act_tanh.forward(z1)
    z2 = dense2.forward(a1)
    a2 = act_tanh.forward(z2)
    z3 = dense3.forward(a2)
    a3 = act_tanh.forward(z3)
    z_out = output.forward(a3)
    y_pred = act_softmax.forward(z_out)

    # Loss calculation
    data_loss = loss.calculate(y_pred, y)
    all_losses.append(data_loss)
    accuracy = loss.accuracy(y, y_pred)
    all_accuracy.append(accuracy)
    # Backward pass
    dLoss_dZ = loss.backward(y_pred, y) # First go through softmax backward
    dLoss_dZ3 = output.backward(dLoss_dZ)  # Then through the dense layer
    dLoss_dA2 = act_tanh.backward(dLoss_dZ3)
    dLoss_dZ2 = dense3.backward(dLoss_dA2)
    dLoss_dA1 = act_tanh.backward(dLoss_dZ2)
    dLoss_dZ1 = dense2.backward(dLoss_dA1)  # We don't need dLoss_dA0 for any further calculations

    # Update weights
    optimizer.update_weights(dense1)
    optimizer.update_weights(dense2)
    optimizer.update_weights(dense3)
    optimizer.update_weights(output)
    optimizer.update_params()

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {data_loss}")

