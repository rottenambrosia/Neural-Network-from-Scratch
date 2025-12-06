from random import gauss

import numpy as np
import sys

class LinearRegression :
    def __init__(self):
        pass

class PolynomialRegression :
    def __init__(self):
        pass

class Dense:
    def __init__(self, n_inputs, n_neurons, activation_type):
        # Gradients
        self.dLoss_dWeights = None
        self.dLoss_dx = None
        self.dLoss_dBias = None

        # Weights init: He for ReLU, Xavier for tanh
        if activation_type == "tanh":
            limit = np.sqrt(6.0 / (n_inputs + n_neurons))
            self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))
        elif activation_type == "relu" or activation_type=="sigmoid" or activation_type=="softmax":  # default: ReLU
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        else :
            print("Invalid activation type")
            sys.exit(0)
        self.bias = np.zeros((1, n_neurons))
        self.output = None
        self.inputs = None

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output =  np.dot(inputs, self.weights) + self.bias
        return self.output
    
    def backward(self, dLoss_dZ):
        # Gradient on parameters
        self.dLoss_dWeights = np.dot(self.inputs.T, dLoss_dZ)
        self.dLoss_dBias = np.sum(dLoss_dZ, axis=0, keepdims=True)
        # Gradient on values
        self.dLoss_dx = np.dot(dLoss_dZ, self.weights.T)
        return self.dLoss_dx

# Activation Classes :
#rectified linear unit
class ReLU:
    def __init__(self):
        # pass
        self.dLoss_dx = None
        self.output = None
    def forward(self ,x):
        self.output = np.maximum( x, 0)
        return self.output
    def backward(self, dLoss_dOutput):
        # mask: 1 where output > 0, else 0
        self.dLoss_dx = dLoss_dOutput * (self.output > 0)
        return self.dLoss_dx
#sigmoid
class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self ,x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    def backward(self, x):
        self.output = x * (1 - x)
        return self.output
#softmax
class Softmax:
    def __init__(self):
        self.dLoss_dz = None
        self.output = None

    def forward(self ,x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    """mathematically calculating softmax backprop involves finding the Jacobian Matrix with help of the Kronecker delta
            but in practice, that entire derivation just resolves to the difference b/w the true values and the predictions
            the process involves taking each single output, flattening it and building a diagonal matrix from it and
            subtracting the product of the output matrix and its transpose from the diagonal matrix to get the jacobian matrix.
            After that we just multiply the Jacobian Matrix with the dActivation_dz to get dLoss_dz."""
    # def backward(self, dActivation_dz):
    #     self.dLoss_dz = np.zeros(dActivation_dz.shape)
    #     for i in range(len(dActivation_dz)):
    #         self.dLoss_dz[i] = np.diag(dActivation_dz[i])
    #         self.dLoss_dz[i] -= np.dot(dActivation_dz[i], dActivation_dz[i].T) # Jacobian Matrix calculation
    #         self.dLoss_dz[i] /= dActivation_dz[i].shape[0]
    #     return self.dLoss_dz
    """ We also wont be keeping the backward func for the softmax class as we're already calculating it as part of our
    collective loss function"""
    # def backward(self, y_true):
    #     self.dLoss_dz = (y_true - self.output) / y_true.shape[0]
    #     return self.dLoss_dz
#Hyperbolic Tangent activation
class tanh:
    def __init__(self):
        self.dLoss_dx = None
        self.output = None

    def forward(self ,x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, dLoss_dOutput):
        # derivative of tanh: 1 - output^2
        grad = 1 - self.output ** 2
        self.dLoss_dx = dLoss_dOutput * (grad + 1e-7)
        return self.dLoss_dx


# Losses
class Loss :
    def calculate (self, output, y) :
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# Mean Squared Error for regression problems
class MSE_loss:
    @staticmethod
    def forward(y_true, y_pred):
        loss = np.mean((y_true - y_pred) ** 2, axis=-1)
        mean_loss = np.mean(loss)
        return mean_loss
    @staticmethod
    def backward(y_pred, y_true):
        # samples = y_true.shape[0]
        outputs = y_true.shape[1]
        dLoss = -2 * (y_true - y_pred) / outputs
        # dLoss = dLoss / samples
        return dLoss


#Binary Crossentropy for binary classification problems
class BinaryCrossentropy(Loss):
    @staticmethod
    def forward(y_pred, y_true):
        np.clip(y_pred, 1e-7, 1 - 1e-7)
        epsilon = - y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred)
        return epsilon
    @staticmethod
    def backward(y_true, y_pred):
        np.clip(y_pred, 1e-7, 1 - 1e-7)
        dLoss = - y_true/y_pred + (1-y_true)/(1-y_pred)
        return dLoss


# Categorical Crossentropy for multi class classification problems
class CategoricalCrossentropy(Loss):
    def __init__(self):
        self.dLoss_dz = None

    @staticmethod
    def forward(y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = None
        if len(y_true.shape) == 1: # class targets are just numbers
            correct_confidences = y_pred_clipped[
                range(samples), 
                y_true]
        elif len(y_true.shape) == 2: # one-hot encoded class targets
            correct_confidences = np.sum(
                y_true.T * y_pred_clipped, 
                axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # @staticmethod
    def backward(self, dLoss_dActivationOutput, y_true):
        samples = len(dLoss_dActivationOutput)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dLoss_dz = dLoss_dActivationOutput.copy()
        self.dLoss_dz[range(samples), y_true] -= 1
        self.dLoss_dz = self.dLoss_dz / samples
    def accuracy(self,y_true, y_pred):
       predictions = np.argmax(y_pred, axis=1)
       accuracy = np.mean(predictions == y_true)
       return accuracy

# having the backward pass calculations done as part of the same class is about 7 times faster than calculating them in two different steps
class SoftmaxCategoricalCrossentropy (Loss) : #SparseCategoricalCrossentropy in Tensorflow, directly calculating dLoss_dz
    def __init__(self):
        self.dLoss_dZ = None
        self.output = None
        self.activation = Softmax()
        self.loss = CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dLoss_dSoftmax, y_true):
        samples = dLoss_dSoftmax.shape[0]
        if len(y_true.shape)==2 :
            y_true = np.argmax(y_true, axis=1) # converting one-hot encoded data to numbers
        self.dLoss_dZ = dLoss_dSoftmax.copy()
        self.dLoss_dZ[np.arange(samples), y_true] -= 1 #building the confidence matrix and then subtracting the true values (1, since all are the predicted classes) to calculate the gradient
        self.dLoss_dZ = self.dLoss_dZ / samples # normalizing to prevent gradient explosion
        return self.dLoss_dZ
    @staticmethod
    def accuracy(y_true, y_pred):

        predictions = np.argmax(y_pred, axis=1)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == y_true)
        return accuracy

# --- Optimizer ---
# Optimizer with learning rate decay
class GradientDescentOptimizerWithMomentum :
    def __init__(self, learning_rate=1., decay = 0., momentum = 0.):
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    def pre_update_params(self) :
        if self.decay :
            self.current_learning_rate = self.learning_rate / (1+self.decay*self.iterations)
    def update_weights(self, layer):
        if self.momentum :
            if not hasattr(layer, 'weight_momentum'):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.bias)
            weight_updates = (self.momentum * layer.weight_momentum -
                              self.current_learning_rate * layer.dLoss_dWeights)
            layer.weight_momentum = weight_updates
            bias_updates = (self.momentum * layer.bias_momentum -
                            self.current_learning_rate * layer.dLoss_dBias)
            layer.bias_momentum = bias_updates
        else :
            weight_updates = -self.current_learning_rate * layer.dLoss_dWeights
            bias_updates = -self.current_learning_rate * layer.dLoss_dBias
        layer.weights += weight_updates
        layer.bias += bias_updates
    def update_params(self):
        self.iterations += 1

class GradientDescentOptimizerWithRMSProp :
    def __init__(self, learning_rate=1., decay = 0., rho = 0.):
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.rho = rho
    def pre_update_params(self) :
        if self.decay :
            self.current_learning_rate = self.learning_rate / (1+self.decay*self.iterations)
    def update_weights(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dLoss_dWeights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dLoss_dBias**2
        layer.weights += -self.current_learning_rate * layer.dLoss_dWeights / (np.sqrt(layer.weight_cache) + 1e-7)
        layer.bias += -self.current_learning_rate * layer.dLoss_dBias / (np.sqrt(layer.bias_cache) + 1e-7)
    def update_params(self):
        self.iterations += 1    

class AdamOptimizer:
    def __init__(self, learning_rate=1., decay=0., momentum=0., rho=0.):
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum  # beta1
        self.rho = rho            # beta2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

    def update_weights(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.bias)
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.bias)

        # First moment (momentum) 
        layer.weight_momentum = self.momentum * layer.weight_momentum + (1 - self.momentum) * layer.dLoss_dWeights
        layer.bias_momentum = self.momentum * layer.bias_momentum + (1 - self.momentum) * layer.dLoss_dBias

        # Second moment (RMSProp) 
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dLoss_dWeights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dLoss_dBias**2

        # Bias correction
        weight_momentum_corrected = layer.weight_momentum / (1 - self.momentum ** (self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentum / (1 - self.momentum ** (self.iterations + 1))
        weight_cache_corrected = layer.weight_cache / (1 - self.rho ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.rho ** (self.iterations + 1))

        # Update parameters
        layer.weights += -self.current_learning_rate * weight_momentum_corrected / (np.sqrt(weight_cache_corrected) + 1e-7)
        layer.bias += -self.current_learning_rate * bias_momentum_corrected / (np.sqrt(bias_cache_corrected) + 1e-7)

    def update_params(self):
        self.iterations += 1
