from functools import reduce

import torch
from torch import nn, autograd
import numpy as np
from pprint import pprint

from .base import ModelBaseClass
from .utils import DataGenerator, sample_from, counts
from .torch_circuit import (
    ParallelRYComplex, 
    EntangleComplex, 
    Exp, 
    batch_kronecker_complex,
)
from utils import epsilon, ints_to_onehot, evaluate


class Encoder(nn.Module):

    def __init__(self, z_dim: int, data_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Embedding(num_embeddings=2 ** data_dim, embedding_dim=z_dim),
            nn.Linear(z_dim, z_dim),
            nn.LeakyReLU(),
        )
        self.theta_linear = nn.Linear(z_dim, z_dim)
        self.phi_linear = nn.Linear(z_dim, z_dim)
        self.log_kappa_linear = nn.Linear(z_dim, z_dim)

    def forward(self, x):
        x = self.layers(x)
        return (
            torch.sigmoid(self.theta_linear(x)) * np.pi, 
            torch.sigmoid(self.phi_linear(x)) * 2 * np.pi,
        )



class Decoder(nn.Module):

    def __init__(self, n_qubit: int, k: int):
        super().__init__()
        self.preparation_layer = ParallelRYComplex(n_qubit)
        self.layers = nn.ModuleList()
        for _ in range(k):
            self.layers.append(EntangleComplex(n_qubit))
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


class QAE(ModelBaseClass):

    def __init__(self, n_qubit: int, batch_size: int, n_epoch: int, circuit_depth: int, lr: float, **kwargs):
        self.n_qubit = n_qubit
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.encoder = Encoder(n_qubit, n_qubit).to(self.device)
        self.decoder = Decoder(n_qubit, k=circuit_depth).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )

    def fit(self, data: np.array) -> np.array:
        
        data_dist = counts(data, self.n_qubit)
        epoch_div_history = []

        for i_epoch in range(self.n_epoch):
            recon_losses = []
            purity_losses = []
            for real_batch in DataGenerator(data, self.batch_size):
                recon_loss, purity_loss = self.fit_batch(real_batch)
                recon_losses.append(recon_loss)
                purity_losses.append(purity_loss)

            epoch_div_history.append(evaluate(data_dist, self.output_dist(data)))
            if (i_epoch+1) % 5 == 0:
                print(f'epoch{i_epoch+1:3d} RECON: {np.mean(recon_losses):4f} Avg_purity: {-np.mean(purity_losses)/self.n_qubit:4f}')
                print(epoch_div_history[-1])

        kl_div = [x['kl'] for x in epoch_div_history]
        js_div = [x['js'] for x in epoch_div_history]
        return kl_div, js_div

    def get_outcome(self, data: np.array):
        with torch.no_grad():
            z = self.calculate_latent(data)
            probs = self.decoder.forward_prob(z)[0]
        return probs.cpu().data.numpy()

    def output_dist(self, data: np.array):
        probs = []
        for batch in DataGenerator(data, self.batch_size):
            batch = torch.from_numpy(batch).to(self.device).long()
            theta, phi = self.encoder(batch)  # (N, self.n_qubit)
            z = self.prepare_state(theta, phi)
            p = self.decoder.forward_prob(z)
            probs.append(p.mean(dim=0).cpu().data.numpy())
        
        return np.mean(probs, axis=0)

    @staticmethod
    def prepare_state(theta: torch.Tensor, phi: torch.Tensor):
        x_real = torch.cos(theta / 2)
        x_imag = torch.zeros_like(x_real)
        y_real = torch.sin(theta / 2) * torch.cos(phi)
        y_imag = torch.sin(theta / 2) * torch.sin(phi)

        single_qubit_states_real = torch.cat([
            x_real.transpose(0, 1).unsqueeze(-1),
            y_real.transpose(0, 1).unsqueeze(-1),
        ], dim=-1)  # (n_qubit, N, 2)
        single_qubit_states_imag = torch.cat([
            x_imag.transpose(0, 1).unsqueeze(-1),
            y_imag.transpose(0, 1).unsqueeze(-1),
        ], dim=-1)
        
        z = reduce(
            lambda x, y: batch_kronecker_complex(x, y), 
            zip(single_qubit_states_real, single_qubit_states_imag)
        )  # Tuple[(N, 2 ** n_qubit)]
        return z
        

    def fit_batch(self, batch: np.array):
        batch = torch.from_numpy(batch).to(self.device).long()
        theta, phi = self.encoder(batch)  # (N, n_qubit)
        z = self.prepare_state(theta, phi)  
        recon = self.decoder.forward_prob(z)
        recon_loss = -torch.log(
            torch.gather(recon, dim=1, index=batch.unsqueeze(-1)) # inner product, should be a swap test
            + epsilon
        )  
        
        theta, phi = theta.unsqueeze(-1), phi.unsqueeze(-1)
        rho_real = torch.cat([
            torch.cos(theta / 2) ** 2, torch.sin(theta) * torch.cos(phi) / 2,
            torch.sin(theta) * torch.cos(phi) / 2, torch.sin(theta / 2) ** 2,
        ], dim=-1).view(-1, self.n_qubit, 2, 2)
        rho_imag = torch.cat([
            torch.zeros_like(theta), torch.sin(theta) * torch.sin(phi) / 2,
            -torch.sin(theta) * torch.sin(phi) / 2, torch.zeros_like(theta),
        ], dim=-1).view(-1, self.n_qubit, 2, 2)
        
        mean_rho_real, mean_rho_imag = rho_real.mean(dim=0), rho_imag.mean(dim=0)
        purity_loss = self.calculate_purity_loss(mean_rho_real, mean_rho_imag)        
        
        loss = recon_loss.mean() + 100. * purity_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return recon_loss.mean().item(), purity_loss.item()

    def calculate_purity_loss(self, mean_rho_real, mean_rho_imag):

        # purity = tr(rho^2) = | rho_00 |^2 + | rho_01 |^2 + | rho_10 |^2 + | rho_11 |^2

        purity_loss = -(mean_rho_real ** 2 + mean_rho_imag ** 2).sum()
        return purity_loss

    def calculate_latent(self, data: np.array):
        rho_reals, rho_imags, probs = [], [], []
        for batch in DataGenerator(data, self.batch_size):
            batch = torch.from_numpy(batch).to(self.device).long()
            theta, phi = self.encoder(batch)  # (N, self.n_qubit)
            z = self.prepare_state(theta, phi)

            theta = theta.unsqueeze(-1)
            phi = phi.unsqueeze(-1)

            rho_real = torch.cat([
                torch.cos(theta / 2) ** 2, torch.sin(theta) * torch.cos(phi) / 2,
                torch.sin(theta) * torch.cos(phi) / 2, torch.sin(theta / 2) ** 2,
            ], dim=-1).view(-1, self.n_qubit, 2, 2)
            rho_imag = torch.cat([
                torch.zeros_like(theta), torch.sin(theta) * torch.sin(phi) / 2,
                -torch.sin(theta) * torch.sin(phi) / 2, torch.zeros_like(theta),
            ], dim=-1).view(-1, self.n_qubit, 2, 2)
            rho_reals.append(rho_real.mean(dim=0).cpu().data.numpy())
            rho_imags.append(rho_imag.mean(dim=0).cpu().data.numpy())

            p = self.decoder.forward_prob(z)
            probs.append(p.mean(dim=0).cpu().data.numpy())

        probs = np.mean(probs, axis=0)
        mean_rho_real = np.mean(rho_reals, axis=0)
        mean_rho_imag = np.mean(rho_imags, axis=0)

        theta = np.arccos(np.sqrt(mean_rho_real[:, 0, 0])) * 2
        phi = np.arctan(mean_rho_imag[:, 0, 1] / mean_rho_real[:, 0, 1])
        
        theta = torch.from_numpy(theta).unsqueeze(0).to(self.device)
        phi = torch.from_numpy(phi).unsqueeze(0).to(self.device)
        z = self.prepare_state(theta, phi)

        print(
            'purity_loss: ', 
            self.calculate_purity_loss(
                torch.from_numpy(mean_rho_real),
                torch.from_numpy(mean_rho_imag),
            ).item()
        )
        return z