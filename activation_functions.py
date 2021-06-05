import numpy as np

# all derivatives assume that x is already an output of the

def identity(x, derivative=False):
    if not derivative:
        return x
    else:
        return 1

def sigmoid(x, derivative=False):
    if not derivative:
        return 1/(1+np.exp(-x))
    else:
        return x * (1 - x)

def tanh(x, derivative=False):
    if not derivative:
        return 1 - (x^2)
    else:
        return (2/(1+np.exp(-2*x)))-1


def relu(x, derivative=False):
    if not derivative:
        return max(x, 0)
    else:
        return 0 if x < 0 else 1
