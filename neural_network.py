import utils
import nnfs
import nnfs.datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("dark_background")

nnfs.init()

X, y = nnfs.datasets.vertical_data(1000, 3)

fig = plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg", marker="*")
plt.title("Spiral Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("plots/spiral_data.png")
plt.close()

# --- Pipeline ---
layer_1_dense = utils.Dense(2,   256, "relu")
layer_1_activation = utils.ReLU()

layer_2_dense = utils.Dense(256, 256, "relu")
layer_2_activation = utils.ReLU()

layer_3_dense = utils.Dense(256, 256, "relu")
layer_3_activation = utils.ReLU()

layer_4_dense = utils.Dense(256, 256, "relu")  # final logits
layer_4_activation = utils.ReLU()

layer_5_dense = utils.Dense(256, 3, "softmax")   # final softmax
layer_5_activation = utils.Softmax()

layers = [layer_1_dense, layer_2_dense, layer_3_dense, layer_4_dense, layer_5_dense]
loss_function = utils.SoftmaxCategoricalCrossentropy()
optimizer = utils.AdamOptimizer(0.1, 1e-3, 0.9, 0.99)

epochs = 201
all_loss = []
all_accuracy = []

for epoch in range(epochs):
    # ----- forward -----
    layer_1_dense.forward(X)
    layer_1_activation.forward(layer_1_dense.output)

    layer_2_dense.forward(layer_1_activation.output)
    layer_2_activation.forward(layer_2_dense.output)

    layer_3_dense.forward(layer_2_activation.output)
    layer_3_activation.forward(layer_3_dense.output)

    layer_4_dense.forward(layer_3_activation.output)
    layer_4_activation.forward(layer_4_dense.output)

    layer_5_dense.forward(layer_4_activation.output)
    layer_5_activation.forward(layer_5_dense.output)

    # ----- loss -----
    loss = loss_function.forward(layer_5_dense.output, y)
    predictions = layer_5_activation.output
    accuracy = loss_function.accuracy(y, predictions)

    if epoch % 10 == 0:
        print(f"Loss at iteration {epoch} of {epochs}: {loss:.3f}, Accuracy: {accuracy:3f}, learning_rate : {optimizer.current_learning_rate:.3f}")
    all_loss.append(loss)
    all_accuracy.append(accuracy)

    # ----- backward -----
    last_gradients = loss_function.backward(loss_function.output, y)
    layer_5_dense.backward(last_gradients)

    layer_4_activation.backward(layer_5_dense.dLoss_dx)
    layer_4_dense.backward(layer_4_activation.dLoss_dx)

    layer_3_activation.backward(layer_4_dense.dLoss_dx)
    layer_3_dense.backward(layer_3_activation.dLoss_dx)

    layer_2_activation.backward(layer_3_dense.dLoss_dx)
    layer_2_dense.backward(layer_2_activation.dLoss_dx)

    layer_1_activation.backward(layer_2_dense.dLoss_dx)
    layer_1_dense.backward(layer_1_activation.dLoss_dx)

    # ----- update -----
    optimizer.pre_update_params()
    for layer in layers:
        optimizer.update_weights(layer)
    optimizer.update_params()

iterations = np.array(range(len(all_loss)))

plt.figure(figsize=(10, 6))
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title(f"Loss over {epochs-1} Iterations for {len(layers)} Layers, 128 units each:")
plt.plot(iterations, all_loss)
plt.savefig(f"plots/losses/best_loss_over_iterations_{epochs}_layers_{len(layers)}_lr_{optimizer.learning_rate}.png")
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title(f"Accuracy over {epochs-1} Iterations for {len(layers)} Layers, 128 units each:")
plt.plot(iterations, all_accuracy)
plt.axhline(0.55, color='r', linestyle='--', label='Baseline Performance')
plt.savefig(f"plots/accuracies/best_accuracy_over_iterations_{epochs}_layers_{len(layers)}_lr_{optimizer.learning_rate}.png")
plt.show()
plt.close()