import numpy as np
from abc import ABC, abstractmethod
from .tensor import Tensor

class Activation(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

# NOTE: ide dari autodiff, daripada nurunin persamaan yang lumayan kompleks, kita nurunin operasi2 kecilnya
# Misal: Sigmoid: 1 / 1+e^(-x)
# Ubah jadi bentuk: (1 + exp(-x))^-1
# yang diturunin: -x, exp(-x), 1 + exp, ^-1

class Linear(Activation):
    def __call__(self, x: Tensor):
        # we could return x immediately, but to mark the activation with an operation, we have to create a new Node
        out = Tensor(x.data, parents=(x,), op='linear')
        out._backward = lambda: setattr(x, 'grad', x.grad + out.grad)
        return out


class ReLU(Activation):
    def __call__(self, x): 
        return x.relu()

class Sigmoid(Activation):
    def __call__(self, x): 
        return 1 / (1.0 + (-x).exp())
         
class Tanh(Activation):
    def __call__(self, x): 
        return x.tanh()

class Softmax(Activation):
    def __call__(self, x): 
        # we shift to prevent overflow
        shifted_x = x - np.max(x.data, axis=1, keepdims=True)
        exp_x = shifted_x.exp()
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

class SiLU(Activation):
    def __call__(self, x: Tensor):
        # x * sigmoid
        sigmoid_x = 1.0 / (1.0 + (-x).exp())
        return x * sigmoid_x


class Softplus(Activation):
    def __call__(self, x: Tensor):
        return (1.0 + x.exp()).log()
