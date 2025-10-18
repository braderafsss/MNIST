import numpy as np
from collections import OrderedDict

from gradient_2d import numerical_gradient
from layers import Affine, Relu, Sigmoid, SoftmaxWithLoss

class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, activation='relu', weight_init_std='relu', weight_decay_lambda=0.0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda

        self.params = {}
        # initialize weights
        self.__init_weight(weight_init_std)

        # create layers
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        # hidden layers (Affine + Activation)
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                     self.params['b' + str(idx)])
            self.layers['Activation' + str(idx)] = activation_layer[activation]()
        # output layer (last affine)
        last_idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(last_idx)] = Affine(self.params['W' + str(last_idx)],
                                                      self.params['b' + str(last_idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """Initialize weights and biases."""
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            # decide scale
            if isinstance(weight_init_std, str):
                key = weight_init_std.lower()
            else:
                key = None

            if key in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif key in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            else:
                # numeric value or default
                scale = float(weight_init_std)

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """Compute loss (data loss + weight decay)."""
        y = self.predict(x)
        weight_decay = 0.0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """Numerical gradient (slow, for checking)."""
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
        return grads

    def gradient(self, x, t):
        """Backpropagation to compute gradients efficiently."""
        # forward
        self.loss(x, t)

        # backward
        dout = 1.0
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # collect gradients
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            affine_layer = self.layers['Affine' + str(idx)]
            # assume Affine stores dW and db after backward
            grads['W' + str(idx)] = affine_layer.dW + self.weight_decay_lambda * self.params['W' + str(idx)]
            grads['b' + str(idx)] = affine_layer.db
        return grads
