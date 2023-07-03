from functools import partial

import numpy as np
import torch
from torch import nn

from utils import bits_to_ints, epsilon, get_pmf, evaluate
from .base import ModelBaseClass
from .utils import EMA, DataGenerator, sample_from, counts
from .torch_circuit import ParallelRYComplex, ParallelRXComplex, EntangleComplex


class Generator(nn.Module):

    def __init__(self, n_qubit: int, k: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.preparation_layer = nn.Sequential(
            ParallelRXComplex(n_qubit),
            ParallelRYComplex(n_qubit),
            ParallelRXComplex(n_qubit)
        )

        self.layers = nn.ModuleList()
        for _ in range(k):
            self.layers.append(EntangleComplex(n_qubit))
            self.layers.append([
                ParallelRXComplex(n_qubit),
                ParallelRYComplex(n_qubit),
                ParallelRYComplex(n_qubit)
            ])

    def forward(self, x):
        x = self.preparation_layer(x)
        for layer in self.layers:
            x = layer(x)
        probs = x[0] ** 2 + x[1] ** 2
        return probs


class MMD(nn.Module):

    def __init__(self, sigmas: list, n_qubit: int):
        super().__init__()
        self.n_qubit = n_qubit
        self.K = nn.Parameter(self.make_K(sigmas), requires_grad=False)

    def forward(self, x, y):
        x_y = (x - y).unsqueeze(-1)
        return x_y.transpose(0, 1) @ self.K @ x_y

    def to_binary(self, x):
        r = torch.arange(self.n_qubit)
        to_binary_op = torch.ones_like(r) << r  # (n_qubit,)
        return ((x.unsqueeze(-1) & to_binary_op) > 0).long()

    def make_K(self, sigmas: list):
        sigmas = torch.Tensor(sigmas)
        r = self.to_binary(torch.arange(2 ** self.n_qubit)).float()  # (2 ** n_qubit, n_qubit)

        x = r.unsqueeze(1)  # (2 ** n_qubit, 1, n_qubit)
        y = r.unsqueeze(0)  # (1, 2 ** n_qubit, n_qubit)
        
        x_y = torch.einsum('abn,bcn->acn', x, y)  
        norm_square = (x** 2 + y ** 2 - 2 * x * y).sum(dim=-1)  # (2 ** n_qubit, 2 ** n_qubit)
        
        K = (-norm_square.unsqueeze(-1) / (2 * sigmas)).exp().sum(dim=-1)  # (2 ** n_qubit, 2 ** n_qubit)
        return K


class QCBM(ModelBaseClass):

    def __init__(self, n_qubit: int, batch_size: int, n_epoch: int, circuit_depth: int, lr: float, **kwargs):
        self.n_qubit = n_qubit
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.ema = EMA(0.9).to(self.device)
        self.mmd = MMD([0.5, 1., 2., 4.], n_qubit=n_qubit).to(self.device)
        
        self.generator = Generator(self.n_qubit, k=circuit_depth).to(self.device)
        self.optim = torch.optim.Adam(params=self.generator.parameters(), lr=lr)

    def fit(self, data: np.array) -> np.array:
        
        data_dist = counts(data, self.n_qubit)
        epoch_div_history = []
        
        data_pmf = get_pmf(data_dist)
        data_pmf = torch.from_numpy(data_pmf).float().to(self.device)
        for i_epoch in range(self.n_epoch):
            mmd_losses = []
            for _ in DataGenerator(data, self.batch_size):
                mmd_loss = self.train_generator(data_pmf)
                mmd_losses.append(mmd_loss)
            
            epoch_div_history.append(evaluate(data_dist, self.output_dist()))
            if (i_epoch+1) % 5 == 0:
                print(f'epoch: {i_epoch+1} MMD: {np.mean(mmd_losses):4f}', end=' ')
                print(epoch_div_history[-1])
        
        kl_div = [x['kl'] for x in epoch_div_history]
        js_div = [x['js'] for x in epoch_div_history]
        
        return kl_div, js_div

    def output_dist(self):
        z = self.get_prior()
        with torch.no_grad():
            gen_probs = self.generator(z)
        return gen_probs[0].cpu().data.numpy()

    def train_generator(self, data_pmf: torch.Tensor):
        z = self.get_prior().repeat(self.batch_size, 1)
        fake_probs = self.generator(z)

        fake_data = sample_from(fake_probs)      
        selected_probs = torch.gather(fake_probs, dim=-1, index=fake_data)
        log_selected_probs = torch.log(selected_probs + epsilon).squeeze(dim=-1)

        fake_data_pmf = (torch.arange(2 ** self.n_qubit, device=self.device) == fake_data).sum(dim=0).float() / self.batch_size
        
        mmd = self.mmd(data_pmf, fake_data_pmf)
        reward = -mmd
        baseline = self.ema(reward.mean().data)

        loss = (-(reward - baseline) * log_selected_probs).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return mmd.item()

    def get_prior(self) -> torch.Tensor:
        z = torch.zeros([1, 2 ** self.n_qubit]).to(self.device)
        z[:, 0] = 1
        # prepare initial state at "0"
        return z