import re

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from einops import rearrange, reduce, einsum


class RMSNorm(Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = Parameter(torch.ones((d_model,), device=device, dtype=dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(reduce(x ** 2, '... d -> ...', 'mean') + self.eps).to(original_dtype)
        rms = rearrange(rms, '... -> ... 1')
        return x / rms * self.gain