import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import List
from .tensor import Tensor
from .layers import Layer
from .losses import Loss
from .optimizers import SGD, Adam
from src.utils import get_progress_bar

# Adding layers are done manually and modularly after initial construction usinf the add_layer method
class FFNN:
    def __init__(self, loss_function: Loss):
        self.layers: List[Layer] = []   # hidden and outpue
        self.loss_function = loss_function
        self.history = {"train_loss": [], "val_loss": []}
        # TODO: save harus ntimpen bobot

        
    def add_layer(self, layer: Layer):
        self.layers.append(layer)

        
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, lr=0.01, l1=0.0, l2=0.0, verbose=1, optimizer="sgd",
              beta1=0.9, beta2=0.999, epsilon=1e-8  
              ):
        
        params = []
        for layer in self.layers:
            if layer.weights: params.append(layer.weights)
            if layer.bias: params.append(layer.bias)
            if layer.normalization:
                if layer.normalization.learnable: params.append(layer.normalization.learnable)

        opt = None
        match optimizer:
            case "sgd":
                opt = SGD(params, lr)
            case "adam":
                opt = Adam(params, lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
            case _:
                print("[TRAIN ERROR] Unknown Optimizer: ", optimizer)

        for i in range(epochs):
            # data shuffling to avoid sequence learning
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            epoch_loss = 0
            for j in range(0, len(X_train), batch_size):
                batch_X = X_train[j : j + batch_size]
                batch_y = y_train[j : j + batch_size]

                inputs = Tensor(batch_X)
                y_actual = Tensor(batch_y)

                # Forward pass
                out = inputs
                for layer in self.layers:
                    out = layer.forward(out)
                pred: Tensor = out 

                # Loss calculation
                loss_node: Tensor = self.loss_function(y_actual, pred)

                # Regularization
                if l1 > 0.0:
                    l1_penalty = None
                    for layer in self.layers:
                        penalty = layer.weights.abs().sum()
                        l1_penalty = penalty if l1_penalty is None else l1_penalty + penalty

                    if l1 is not None:
                        loss_node = loss_node + l1 * 0.5 * l1_penalty
                    
                if l2 > 0.0:
                    l2_penalty = None
                    for layer in self.layers:
                        penalty = (layer.weights ** 2).sum()
                        l2_penalty = penalty if l2_penalty is None else l2_penalty + penalty

                    if l2 is not None:
                        loss_node = loss_node + l2 * 0.5 * l2_penalty
                        
                opt.zero_out_grad() # to 'gorget' errors
                loss_node.backward()
                opt.step()
                epoch_loss += loss_node.data

            # history storage and validations
            avg_train_loss = epoch_loss / (len(X_train) / batch_size)
            self.history["train_loss"].append(avg_train_loss)
 
            val_loss = 0
            if X_val is not None and y_val is not None:
                v_inputs, v_target = Tensor(X_val), Tensor(y_val)
                v_out = v_inputs
                for layer in self.layers:
                    v_out = layer.forward(v_out)
                val_loss = self.loss_function(v_target, v_out).data
            self.history['val_loss'].append(val_loss)
        

            progress = (i+1)/epochs
            bar = get_progress_bar(progress)
            status = f"Epoch {i+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}"
            if verbose:
                print(f"\r{status} {bar}{progress*100:3.0f}%", end="")
        print()  
        return self.history
        

        
    def predict(self, X: Tensor) -> Tensor:
        prev_out: Tensor = Tensor(X)
        for layer in self.layers:
            prev_out = layer.forward(prev_out)
        return prev_out
        
    def save(self, path: str):
        state = {
            'num_layers': len(self.layers),
            'layer_configs': [
                {
                    'input_size': layer.input_size,
                    'output_size': layer.output_size,
                    'activation': layer.activation.__class__.__name__,
                    'weights': layer.weights.data,
                    'bias': layer.bias.data,
                }
                for layer in self.layers
            ],
            'loss_function': self.loss_function.__class__.__name__,
            'history': self.history,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str):
        from .activations import Linear, ReLU, Sigmoid, Tanh, Softmax, SiLU, Softplus
        from .losses import MSE, BinaryCrossEntropy, CategoricalCrossEntropy
        from .layers import Layer

        activation_map = {
            'Linear': Linear, 'ReLU': ReLU, 'Sigmoid': Sigmoid,
            'Tanh': Tanh, 'Softmax': Softmax, 'SiLU': SiLU, 'Softplus': Softplus
        }
        loss_map = {
            'MSE': MSE, 'BinaryCrossEntropy': BinaryCrossEntropy,
            'CategoricalCrossEntropy': CategoricalCrossEntropy
        }

        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.layers = []
        self.history = state['history']
        self.loss_function = loss_map[state['loss_function']]()

        for cfg in state['layer_configs']:
            act = activation_map[cfg['activation']]()
            layer = Layer(cfg['input_size'], cfg['output_size'], act)
            layer.weights = Tensor(cfg['weights'])
            layer.bias = Tensor(cfg['bias'])
            self.layers.append(layer)


    def plot_weights(self, layer_indices: List[int]):
        fig, axes = plt.subplots(1, len(layer_indices), figsize=(5 * len(layer_indices), 4))
        if len(layer_indices) == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, layer_indices):
            weights = self.layers[idx].weights.data.flatten()
            ax.hist(weights, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'Layer {idx} - Weight Distribution')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

    def plot_gradients(self, layer_indices: List[int]):
        fig, axes = plt.subplots(1, len(layer_indices), figsize=(5 * len(layer_indices), 4))
        if len(layer_indices) == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, layer_indices):
            grads = self.layers[idx].weights.grad.flatten()
            ax.hist(grads, bins=50, color='tomato', edgecolor='black', alpha=0.7)
            ax.set_title(f'Layer {idx} - Gradient Distribution')
            ax.set_xlabel('Gradient Value')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

    def plot_learning_curve(self):
        if not self.history['train_loss']:
            print("[PLOT ERROR] No training history found. Train the model first!")
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(self.history['train_loss'], label='Training Loss', color='royalblue', linewidth=2)
        
        if any(self.history['val_loss']): 
            ax.plot(self.history['val_loss'], label='Validation Loss', color='darkorange', linewidth=2)
        
        ax.set_title('Model Learning Curve')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
