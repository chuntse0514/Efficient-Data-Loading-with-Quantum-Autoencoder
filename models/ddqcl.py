from functools import reduce

import torch
from torch import nn, autograd
import numpy as np
from pprint import pprint

from .base import ModelBaseClass
from .utils import DataGenerator, sample_from, counts
from .torch_circuit import (
    ParallelRXComplex,
    ParallelRYComplex, 
    EntangleComplex, 
    Mølmer_Sørensen_XX_layer,
    Exp, 
    batch_kronecker_complex,
)
from utils import epsilon, ints_to_onehot, evaluate


class Decoder(nn.Module):

    def __init__(self, n_qubit: int, k: int):
        super().__init__()
        self.preparation_layer = nn.Sequential(
            ParallelRXComplex(n_qubit),
            ParallelRYComplex(n_qubit),
            ParallelRXComplex(n_qubit)
        )
        self.layers = nn.ModuleList()
        for _ in range(k):
            # self.layers.append(EntangleComplex(n_qubit))
            self.layers.append(Mølmer_Sørensen_XX_layer(n_qubit))
            self.layers.append(ParallelRYComplex(n_qubit))

    def forward_prob(self, x):
        x = self.forward(x)
        probs = x[0] ** 2 + x[1] ** 2
        return probs

    def forward(self, x):
        x = self.preparation_layer(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DDQCL(ModelBaseClass):

    def __init__(self, n_qubit: int, batch_size: int, n_epoch: int, circuit_depth: int, lr: float,**kwargs):
        self.n_qubit = n_qubit
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.decoder = Decoder(n_qubit, k=circuit_depth).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.decoder.parameters()),
            lr=lr
        )

    def fit(self, data: np.array):

        data_dist = counts(data, self.n_qubit)
        epoch_div_history = []

        for i_epoch in range(self.n_epoch):
            nlls = []
            for real_batch in DataGenerator(data, self.batch_size):
                nll = self.fit_batch(real_batch)
                nlls.append(nll)

            epoch_div_history.append(evaluate(data_dist, self.output_dist()))
            if (i_epoch+1) % 5 == 0:
                print(f'epoch: {i_epoch+1:3d} NLL: {np.mean(nlls):4f}', end=' ')
                print(epoch_div_history[-1])
        
        kl_result = [x['kl'] for x in epoch_div_history]
        js_result = [x['js'] for x in epoch_div_history]
        
        return kl_result, js_result

    def output_dist(self):
        with torch.no_grad():
            z = self.prepare_input()
            probs = self.decoder.forward_prob(z)[0]
        return probs.cpu().data.numpy()

    def fit_batch(self, batch: np.array):
        z = self.prepare_input()  
        probs = self.decoder.forward_prob(z)
        probs = probs.repeat(self.batch_size, 1)
        batch = torch.from_numpy(batch).to(self.device).long()
        nll = -torch.log(
            torch.gather(probs, dim=1, index=batch.unsqueeze(-1)) # inner product, should be a swap test
            + epsilon
        )  
        loss = nll.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return nll.mean().item()

    def prepare_input(self):
        z_real = torch.zeros(1, 2 ** self.n_qubit).to(self.device)
        z_real[:, 0] = 1.
        z_imag = torch.zeros_like(z_real)
        return z_real, z_imag
