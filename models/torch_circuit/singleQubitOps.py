from functools import reduce

import numpy as np
import torch
from torch import nn
# from models.torch_circuit import kronecker, kronecker_complex

class RY(nn.Module):
    
    def __init__(self, n_qubit: int, index: int, init_method='zero'):
        super().__init__()
        self.n_qubit = n_qubit
        self.index = index

        if init_method == 'zero':
            self.params = nn.Parameter(torch.zeros(1))
        elif init_method == 'uniform':
            self.params = nn.Parameter((torch.rand(1) * 2 - 1) * np.pi)
        else:
            raise ValueError("init_method must be either 'zero' or 'uniform'!")
        
        self.id1 = nn.Parameter(torch.eye(2 ** index), requires_grad=False)
        self.id2 = nn.Parameter(torch.eye(2 ** (n_qubit - index - 1)), requires_grad=False)

    def forward(self, x: tuple):
        
        cos = torch.cos(self.params)
        sin = torch.sin(self.params)
        ry = torch.cat([cos, -sin, sin, cos], dim=-1).view(2, 2)
        
        op = reduce(lambda a, b: kronecker(a, b), [self.id1, ry, self.id2])

        return (x[0] @ op, x[1] @ op)
    
    def to_matrix(self):
        cos = torch.cos(self.params)
        sin = torch.sin(self.params)
        ry = torch.cat([cos, -sin, sin, cos], dim=-1).view(2, 2)

        return reduce(lambda a, b: kronecker(a, b), [self.id1, ry, self.id2]).detach()
    
class X(nn.Module):

    def __init__(self, n_qubit: int, index: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.index = index
        self.id1 = torch.eye(2 ** index)
        self.id2 = torch.eye(2 ** (n_qubit - index - 1))
        self.x = torch.Tensor([[0, 1], [1, 0]])
        
        self.op = nn.Parameter(reduce(lambda a, b: kronecker(a, b), [self.id1, self.x, self.id2]), requires_grad=False)

    def forward(self, x: tuple):
        return (x[0] @ self.op, x[1] @ self.op)
    
    def to_matrix(self):
        return self.op.data