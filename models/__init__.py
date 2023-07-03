from .gan import GAN
from .qgan import QGAN
from .vae import VAE
from .qae import QAE
from .qcbm import QCBM
from .ddqcl import DDQCL


MODEL_HUB = {
    "GAN": GAN,
    "QGAN": QGAN,
    'VQE': VAE,
    'QAE': QAE,
    'QCBM': QCBM,
    'DDQCL': DDQCL
}
