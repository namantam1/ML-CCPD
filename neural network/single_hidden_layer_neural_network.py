
import numpy as np

class ANN:
    def __init__(self, neurons=5, epochs=2000, lr=0.5):
        self.hidden_layer_neurons = neurons
        self.epochs = epochs
        self.lr = lr

    @staticmethod
    def _sigmoid(k):
        return 1/(1+np.exp(-k))

    @staticmethod
    def _sigmoid_derivative(k):
        return k * (1 - k)

    def fit(self, X, Y):
        inp_layer, out_layer = X.shape[1], Y.shape[1]
        hid_layer = self.hidden_layer_neurons
        epochs, lr = self.epochs, self.lr

        self.hid_weights = np.random.uniform(size=(inp_layer, hid_layer))
        self.hid_bias = np.random.uniform(size=(1, hid_layer))

        self.out_weights = np.random.uniform(size=(hid_layer, out_layer))
        self.out_bias = np.random.uniform(size=(1, out_layer))

        for _ in range(epochs):
            # Forward propogation
            hid_layer_out, predicted_out = self._predict(X)

            # Backward propogation
            error = Y - predicted_out
            d_predicted_output = error * self._sigmoid_derivative(predicted_out)

            error_hid_layer = d_predicted_output.dot(self.out_weights.T)
            d_hid_layer = error_hid_layer * self._sigmoid_derivative(hid_layer_out)

            # updating the weights and bias
            self.out_weights += (hid_layer_out.T @ d_predicted_output) * lr
            self.out_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
            self.hid_weights += (X.T @ d_hid_layer)  * lr
            self.hid_bias += np.sum(d_hid_layer, axis=0, keepdims=True) * lr
        
        return predicted_out


    def _predict(self, X):
        hid_layer_activation = (X @ self.hid_weights) + self.hid_bias
        hid_layer_out = self._sigmoid(hid_layer_activation)

        out_layer_activation = (hid_layer_out @ self.out_weights) + self.out_bias
        predicted_out = self._sigmoid(out_layer_activation)
        return hid_layer_out, predicted_out

    def predict(self, X):
        return self._predict(X)[1]
