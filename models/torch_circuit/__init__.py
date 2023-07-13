from functools import reduce

import numpy as np
import torch
from torch import nn
from typing import Union
from .singleQubitOps import RY, X
from .twoQubitOps import Projector, CRY, CX

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))


def kronecker_complex(A: tuple, B: tuple):
    A_r, A_i = A
    B_r, B_i = B
    return (
        kronecker(A_r, B_r) - kronecker(A_i, B_i),
        kronecker(A_i, B_r) + kronecker(A_r, B_i),
    )

def batch_kronecker(A, B):
    return torch.einsum("na,nb->nab", A, B).view(A.size(0), A.size(1) * B.size(1))


def batch_kronecker_complex(A: tuple, B: tuple):
    A_r, A_i = A
    B_r, B_i = B
    return (
        batch_kronecker(A_r, B_r) - batch_kronecker(A_i, B_i),
        batch_kronecker(A_i, B_r) + batch_kronecker(A_r, B_i),
    )


class ParallelRYComplex(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.params = nn.Parameter(
            (torch.rand([n_qubit]) * 2 - 1) * np.pi
        )

    def forward(self, x: tuple):
        cos = torch.cos(self.params / 2).view(self.n_qubit, 1)
        sin = torch.sin(self.params / 2).view(self.n_qubit, 1)
        single_qubit_gates = torch.cat([cos, -sin, sin, cos], dim=-1).view(self.n_qubit, 2, 2)
        op = reduce(lambda a, b: kronecker(a, b), single_qubit_gates)
        return (x[0] @ op, x[1] @ op)

class ParallelRXComplex(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.params = nn.Parameter(
            (torch.rand([n_qubit]) * 2 - 1) * np.pi
        )

    def forward(self, x: tuple):
        cos = torch.cos(self.params / 2).view(self.n_qubit, 1)
        sin = torch.sin(self.params / 2).view(self.n_qubit, 1)
        zero = torch.zeros_like(cos)

        real_gate = torch.cat([cos, zero, zero, cos], dim=-1).view(self.n_qubit, 2, 2)
        imag_gate = torch.cat([zero, -sin, -sin, zero], dim=-1).view(self.n_qubit, 2, 2)

        op = reduce(lambda a, b: kronecker_complex(a, b), zip(real_gate, imag_gate))
        return (x[0] @ op[0] - x[1] @ op[1], x[0] @ op[1] + x[1] @ op[0])
    
class ParallelRZComplex(nn.Module):
    
    def __init__(self, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.params = nn.Parameter(
            (torch.rand([n_qubit]) * 2 - 1) * np.pi
        )
    
    def forward(self, x: tuple):
        cos = torch.cos(self.params / 2).view(self.n_qubit, 1)
        sin = torch.sin(self.params / 2).view(self.n_qubit, 1)
        zero = torch.zeros_like(cos)

        real_gate = torch.cat([cos, zero, zero, cos], dim=-1).view(self.n_qubit, 2, 2)
        imag_gate = torch.cat([-sin, zero, zero, sin], dim=-1).view(self.n_qubit, 2, 2)

        op = reduce(lambda a, b: kronecker_complex(a, b), zip(real_gate, imag_gate))
        return (x[0] @ op[0] - x[1] @ op[1], x[0] @ op[1] + x[1] @ op[0])

class Mølmer_Sørensen_XX_gate(nn.Module):

    def __init__(self, n_qubit: int, index: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.params = nn.Parameter(
            # (torch.rand(1) * 2 - 1) * np.pi
            torch.zeros(1)
        )
        self.id1_real = nn.Parameter(torch.eye(2 ** index), requires_grad=False)
        self.id1_imag = nn.Parameter(torch.zeros(2 ** index, 2 ** index), requires_grad=False)
        
        self.id2_real = nn.Parameter(torch.eye(2 ** (n_qubit - index - 2)), requires_grad=False)
        self.id2_imag = nn.Parameter(torch.zeros(2 ** (n_qubit - index - 2), 2 ** (n_qubit - index - 2)), requires_grad=False)

    def forward(self, x: tuple):

        cos = torch.cos(self.params)
        sin = torch.sin(self.params)
        zero = torch.zeros_like(cos)

        real_gate = torch.cat([
            cos, zero, zero, zero, \
            zero, cos, zero, zero, \
            zero, zero, cos, zero, \
            zero, zero, zero, cos
        ], dim=-1).view(4, 4)

        imag_gate = torch.cat([
            zero, zero, zero, -sin, \
            zero, zero, -sin, zero, \
            zero, -sin, zero, zero, \
            -sin, zero, zero, zero
        ], dim=-1).view(4, 4)

        complex_gate = (real_gate, imag_gate)
        id1 = (self.id1_real, self.id1_imag)
        id2 = (self.id2_real, self.id2_imag)

        op = kronecker_complex(id1, complex_gate)
        op = kronecker_complex(op, id2)

        return (x[0] @ op[0] - x[1] @ op[1], x[0] @ op[1] + x[1] @ op[0])


class Mølmer_Sørensen_XX_layer(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        
        self.n_qubit = n_qubit
        self.layer = nn.ModuleList()
        for i in range(n_qubit - 1):
            self.layer.append(Mølmer_Sørensen_XX_gate(n_qubit, i))

    def forward(self, x: tuple):
        
        for gate in self.layer:
            x = gate(x)
        return x

class EntangleComplex(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        gates = []
        dim = 2 ** n_qubit
        for i in range(n_qubit):
            c_idx, v_idx = i, (i + 1) % n_qubit
            g = torch.eye(dim)
            for j in range(dim):
                if (j // 2 ** c_idx) % 2 == 1 and (j // 2 ** v_idx) % 2 == 1:
                    g[j, j] = -1            
            gates.append(g)

        self.op = nn.Parameter(reduce(lambda x, y: x * y, gates), requires_grad=False)

    def forward(self, x: torch.Tensor):
        return (x[0] @ self.op, x[1] @ self.op)


class Exp(nn.Module):

    def __init__(self, n_qubit: int):
        super().__init__()
        # comments shape for n_qubit = 3
        r = torch.arange(2 ** n_qubit).long()
        self.m = self.binary(r, n_qubit).float()  # (8, 3)

    def forward(self, x):
        return (torch.abs(x) ** 2) @ self.m

    def binary(self, x, bits):
        mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()