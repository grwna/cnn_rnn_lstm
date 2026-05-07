import numpy as np

from src.ffnn.normalizations import Normalization
from .tensor import Tensor
from .activations import Activation

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: Activation, normalization: Normalization=None):
        self.weights: Tensor = None
        self.bias: Tensor = None
        self.activation: Activation = activation

        self.input_size = input_size
        self.output_size = output_size
        self.normalization = normalization
        

    def initialize_weights(self, method: str, **kwargs):
        """
            methods and arguments: 
               - zero 
               - uniform: seed, lower_bound, upper_bound
               - normal: seed, mean, variance
               - Xavier: seed
               - He: seed
            """

        weight_shape = (self.input_size, self.output_size)
        
        if method == "zero":
            self.weights = Tensor(np.zeros(weight_shape))
            
        elif method == "uniform":
            lb = kwargs.get('lower_bound', -1.0)
            ub = kwargs.get('upper_bound', 1.0)
            seed = kwargs.get('seed', None)

            if seed is not None:
                np.random.seed(seed)

            weights = np.random.uniform(low=lb, high=ub, size=weight_shape)
            self.weights = Tensor(weights)
            
        elif method == "normal":
            mean = kwargs.get('mean', 0.0)
            vari = kwargs.get('variance', 1.0)
            seed = kwargs.get('seed', None)
            
            if seed is not None:
                np.random.seed(seed)

            weights = np.random.normal(loc=mean, scale=np.sqrt(vari), size=weight_shape)
            self.weights = Tensor(weights)
            
        elif method == "xavier":
            # Uniform Xavier Initialization
            # For each weight in network we draw a random value w from a uniform distribution in the range [-limit, limit]
            # Best for Sigmoid and Tanh
            seed = kwargs.get('seed', None)
            if seed is not None:
                np.random.seed(seed)
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            weights = np.random.uniform(-limit, limit, size=weight_shape)
            self.weights = Tensor(weights)

        elif method == "he":
            # Weights are initialized depending on the size of the previous layer which helps in attaining a global minimum of the cost function faster and more efficiently
            # Best for ReLU
            seed = kwargs.get('seed', None)
            if seed is not None:
                np.random.seed(seed)
            std = np.sqrt(2 / self.input_size)
            weights = np.random.normal(0, std, size=weight_shape)
            self.weights = Tensor(weights)
        else:
            print("Weight Initialization: Unknown method!")

        self.bias = Tensor(np.zeros((1, self.output_size)))   # init bias to 0 regardless of weights

    def forward(self, input_data: Tensor) -> Tensor:
        Z = (input_data @ self.weights) + self.bias
        # this is where normalization happens
        if self.normalization:
            Z = self.normalization.normalize(Z)
        
        return self.activation(Z)


