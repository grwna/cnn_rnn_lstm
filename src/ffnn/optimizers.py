from abc import ABC, abstractmethod
from .tensor import Tensor
import numpy as np

class Optimizer(ABC):
    def __init__(self, parameters: list[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self): pass

    def zero_out_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

            
# NOTE: Optimizers only need to update teh weights, since forward pass, loss calculation, and backward pass is already done by autodiff
class SGD(Optimizer):
    # W = W - eta * grad
    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
        
class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
            - lr : stepsize
            - beta1, beta2 : exponential decay rates for the moment estimates
            - m : 1st moment, moving average of gradient -> update direction
            - v : 2nd moment, moving average of gradient squared -> how big update
            - m_hat : bias-corrected 1st moment 
            - v_hat : bias-corrected 2nd moment 
        """
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p.data) for p in self.parameters] # Initialize 1st moment vector
        self.v = [np.zeros_like(p.data) for p in self.parameters] # Initialize 2nd moment vector
        self.t = 0  # time step

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            g = p.grad # get the gradient

            # update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # update weights
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
