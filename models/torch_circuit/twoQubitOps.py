from functools import reduce

import numpy as np
import torch
from torch import nn
# from models.torch_circuit import kronecker, kronecker_complex
# from .singleQubitOps import RY, X
from typing import Union

class Projector(nn.Module):
    
    def __init__(self, n_qubit: int, basis: int, index: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.basis = basis
        self.index = index

        self.id1 = torch.eye(2 ** index)
        self.id2 = torch.eye(2 ** (n_qubit - index - 1))
        self.proj = torch.Tensor([[1, 0], [0, 0]]) if basis == 0 else \
                    torch.Tensor([[0, 0], [0, 1]])
        self.op = nn.Parameter(reduce(lambda a, b: kronecker(a, b), [self.id1, self.proj, self.id2]), requires_grad=False)

    def forward(self, x: tuple):
        return (x[0] @ self.op, x[1] @ self.op)
    
    def to_matrix(self):
        return self.op

class CRY(nn.Module):

    def __init__(self, n_qubit: int, index: Union[tuple, list], init_method='zero'):
        super().__init__()
        self.n_qubit = n_qubit
        self.index = index
        
        self.op1 = Projector(n_qubit, basis=0, index=index[0])
        self.op2 = nn.Sequential(
            Projector(n_qubit, basis=1, index=index[0]),
            RY(n_qubit, index=index[1], init_method=init_method)
        )

    def forward(self, x: tuple):
        x1 = self.op1(x)
        x2 = self.op2(x)
        return (x1[0] + x2[0], x1[1] + x2[1])
    
    def to_matrix(self):
        id_real = torch.eye(2 ** self.n_qubit)
        id_imag = torch.zeros_like(id_real)
        return self.forward((id_real, id_imag))[0].detach()

class CX(nn.Module):
    
    def __init__(self, n_qubit: int, index: Union[tuple, list]):
        super().__init__()
        self.n_qubit = n_qubit
        self.op1 = Projector(n_qubit, basis=0, index=index[0])
        self.op2 = nn.Sequential(
            Projector(n_qubit, basis=1, index=index[0]),
            X(n_qubit, index=index[1])
        )

    def forward(self, x: tuple): 
        x1 = self.op1(x)
        x2 = self.op2(x)
        return (x1[0] + x2[0], x1[1] + x2[1])

    def to_matrix(self):
        id_real = torch.eye(2 ** self.n_qubit)
        id_imag = torch.zeros_like(id_real)
        return self.forward((id_real, id_imag))[0].detach()