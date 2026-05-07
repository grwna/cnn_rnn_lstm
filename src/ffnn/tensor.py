from __future__ import annotations
import numpy as np

# NOTE: kalo mau nambah activation, kemungkinan harus nambah operasi disini
class TensorCalculations:
    def _to_tensor(self, tensor):
        return tensor if isinstance(tensor, Tensor) else Tensor(tensor)

    # makes tensors a certain dimesnions (backward broadcast)
    def _broadcast(self, rhs, target_shape):
        """
        rhs: represents right hand operand (tensor)
        """
        if rhs.shape == target_shape:
            return rhs
        
        # rhs is larger
        while rhs.ndim > len(target_shape):
            rhs = rhs.sum(axis=0)
        
        # contains 1 dims
        for i, dim in enumerate(target_shape):
            if dim == 1 and rhs.shape[i] != 1:
                rhs = rhs.sum(axis=i, keepdims=True)

        return rhs
        
    
    # NOTE: other is the "other" operand
    # TODO: ini mungkin bisa di modularin biar kodenya singkat
    # overloaded operators 
    def __add__(self, other):
        other = self._to_tensor(other)

        # create new child node, define and assign the backward function
        out = Tensor(self.data + other.data, parents=(self, other), op='+')

        def _backward():
            # Chain Rule
            # use the form 'z = x + y', and L for 'Loss'
            # dL/dx (self.grad) = dL/dz (out.grad) * dz/dx (1)
            # also do this for y

            self.grad += self._broadcast(out.grad, self.data.shape) # this should be multiplied by one here, removed for efficiency
            other.grad += self._broadcast(out.grad, other.data.shape)
        
        out._backward = _backward
        return out

    
    def __mul__(self, other): 
        """
        Matrix Hadamard multiplication 
        """
        other = self._to_tensor(other)

        # create new child node, define and assign the backward function
        out = Tensor(self.data * other.data, parents=(self, other), op='*')

        def _backward():
            # Chain Rule
            # use the form 'z = x * y', and L for 'Loss'
            # dL/dx (self.grad) = dL/dz (out.grad) * dz/dx (y)
            # also do this for y

            self.grad += self._broadcast(out.grad * other.data, self.data.shape) 
            other.grad += self._broadcast(out.grad * self.data, other.data.shape)
        
        out._backward = _backward
        return out
        
    
    def __pow__(self, other): 
        """
        other: scalar exponent
        """
        # create new child node, define and assign the backward function
        out = Tensor(self.data **  other, parents=(self,), op='^')

        def _backward():
            # Chain Rule
            # use the form 'z = x^n', and L for 'Loss'
            # dL/dx (self.grad) = dL/dz (out.grad) * dz/dx
            # dz/dx = n * x^(n-1)

            # no broadcast required
            base = np.sign(self.data) * np.maximum(np.abs(self.data), 1e-15) # ensure no 0^-x happens, no div by zero
            self.grad += out.grad * (other * (base ** (other -1 )))
        
        out._backward = _backward
        return out

    
    def __neg__(self):
        out = Tensor(-self.data, parents=(self,), op="-")

        def _backward():
            # d(-x)/dx = -1
            self.grad += -out.grad
            
        out._backward = _backward
        return out 

    def __matmul__(self, other): 
        other = self._to_tensor(other)
        # A @ B
        out = Tensor(self.data @ other.data, parents=(self, other), op='@')

        def _backward():
            # multiply with adjoint to preserve dimensions
            # dL/dA = dL/dout @ B.T
            self.grad += out.grad @ other.data.T
            # dL/dB = A.T @ dL/dout 
            other.grad += self.data.T @ out.grad
        
        out._backward = _backward
        return out
        
    
    # ========== Wrapper Overload Operations ============
    def __sub__(self, other):
        return self + (-other)
        
    def __rsub__(self, other):
        return self._to_tensor(other) + (-self)
        
    def __truediv__(self, other):
       return self * (other**-1) 
        
        
    def __rtruediv__(self, other):
       return self._to_tensor(other) * (self**-1) 

    def __radd__(self, other):
        return self._to_tensor(other) + self
    
    def __rmul__(self, other):
        return self._to_tensor(other) * self
        
        
    # ========== NON Overloaded Ooperations ============
    #
    # reductions/aggregations for loss functions
    #
    def sum(self, axis=None, keepdims=False): 
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), parents=(self,), op="sum")

        def _backward():
            grad_reshaped = out.grad
            # restore removed dimensions
            if axis is not None and not keepdims:
                shape = list(self.data.shape)
                if isinstance(axis, int):
                    shape[axis] = 1
                else:
                    for ax in axis:
                        shape[ax] = 1
                grad_reshaped = out.grad.reshape(shape)

            # dL/dInput = dL/dSum * 1
            self.grad += grad_reshaped
            
        out._backward = _backward
        return out
        
    def mean(self, axis=None, keepdims=False, epsilon=1e-8): 
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), parents=(self,), op="mean")

        def _backward():
            grad_reshaped = out.grad
            # restore removed dimensions
            if axis is not None and not keepdims:
                shape = list(self.data.shape)
                if isinstance(axis, int):
                    shape[axis] = 1
                else:
                    for ax in axis:
                        shape[ax] = 1
                grad_reshaped = out.grad.reshape(shape)

            # dL/dInput = dL/dMean / N
            # N is for local derivative (1/N is local derivative)
            N = self.data.size / out.data.size
            self.grad += grad_reshaped / N
        
        out._backward = _backward
        return out
    
    #
    # primitives for activations/losses
    #
    def exp(self):
        out = Tensor(np.exp(self.data), parents=(self,), op="exp")

        def _backward():
            # dL/dx = dL/d(exp(x)) * d(exp(x))/dx
            # d(exp(x))/dx = exp(x)
            self.grad += out.grad * out.data
            
        out._backward = _backward
        return out

    def log(self):
        # ln (inverse of exp)
        clipped = np.maximum(self.data, 1e-15) # ln for <= 0 is undefined
        out = Tensor(np.log(clipped), parents=(self,), op="log")

        def _backward():
            # dz/dx = 1/x
            self.grad += out.grad * (1/clipped)
        
        
        out._backward = _backward
        return out

    
    def relu(self):
        # NOTE: pake ini buat activation nanti
        out = Tensor(np.maximum(0, self.data), parents=(self,), op="ReLU")

        def _backward():
            # dz/dx of ReLU = 1 if x > 0, else 0 if x < 0
            self.grad += out.grad * (self.data > 0)
        
        out._backward = _backward
        return out
        

    def tanh(self):
        out = Tensor(np.tanh(self.data), parents=(self,), op="tanh")

        def _backward():
            # dz/dx of tanh = 1 - tanh^2(x)
            self.grad += out.grad * (1 - out.data**2)
        
        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), parents=(self,), op="abs")

        def _backward():
            self.grad += out.grad * np.sign(self.data)

        out._backward = _backward
        return out
        

class Tensor(TensorCalculations):
    """
    Represents a single node in computational graph, holding NP array
    """

    def __init__(self, data, parents=(), op=''):
        # type error protection
        if isinstance(data, (np.ndarray, list, float, int)):
            self.data = np.array(data, dtype=np.float64)
        elif hasattr(data, 'data'):
            self.data = np.array(data.data, dtype=np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.grad = np.zeros_like(self.data)     # gradient of final output wrt to this specific node
        self._backward = lambda: None   # for each nodes, thsi will hold the chain rule for that specific node, which will be dynamically changing as the backward pass progresses
        
        # NOTE: the parents are what makes up the operation that results in this node, however, in a computational graphs
        # the leaf are "parents" while the root is the final child.
        self.parents = set(parents) 
        
        self.op = op      # the operation that produced this node

    

        
    # recursive helper
    def _topo_sort(self, node: Tensor, visited: set, topo: list):
        if node in visited:
            return
        
        visited.add(node)
        for parent in node.parents:
            self._topo_sort(parent, visited, topo)
        topo.append(node)
        
        
    def backward(self):
        # Loss: dL/dL = 1
        self.grad = np.ones_like(self.data)

        visited = set()
        topo = list()
        self._topo_sort(self, visited, topo)

        for node in reversed(topo):
            node._backward()
        
        
