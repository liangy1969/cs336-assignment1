from math import e

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from einops import einsum

class Linear(Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        param_std = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=param_std, a=-3.0 * param_std, b=3.0 * param_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in, out in -> ... out")
    

class SwiGLU(Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        weight_std = (2 / (d_model + d_ff)) ** 0.5
        # weight for silu activation
        self.weight_1 = Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight_1, mean=0.0, std=weight_std, a=-3.0 * weight_std, b=3.0 * weight_std)
        # weight for output projection
        self.weight_2 = Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight_2, mean=0.0, std=weight_std, a=-3.0 * weight_std, b=3.0 * weight_std)
        # weight for input projection
        self.weight_3 = Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight_3, mean=0.0, std=weight_std, a=-3.0 * weight_std, b=3.0 * weight_std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_projection = einsum(x, self.weight_3, "... d, ff d -> ... ff")
        silu_projection = einsum(x, self.weight_1, "... d, ff d -> ... ff")
        silu_activation = silu_projection * torch.sigmoid(silu_projection)
        gated_projection = input_projection * silu_activation
        output_projection = einsum(gated_projection, self.weight_2, "... ff, d ff -> ... d")
        return output_projection