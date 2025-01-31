import torch
import torch.nn as nn
from .base_autoencoder import BaseAutoencoder

class PolyFit(BaseAutoencoder):
    """
    Baseline model based on a second-degree polynomial parameterized with three learnable coefficients.
    """
    def build(self):
        self.coefficients = nn.Parameter(torch.tensor([1, 0.0, 0.0]))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # first value of the sequence
        x0 = x[:, 0, :].unsqueeze(1)  # [batch_size, 1, 1]
        
        # time index
        t = torch.linspace(0, 1, seq_len).to(x.device).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]

        output = x0 + (
            self.coefficients[0]  # Constant term
            + self.coefficients[1] * t  # Linear term
            + self.coefficients[2] * t ** 2  # Quadratic term
        )
        return output