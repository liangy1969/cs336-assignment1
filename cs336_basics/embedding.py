import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from einops import rearrange


class Embedding(Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding_table = Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.embedding_table, mean=0.0, std=1.0, a=-3, b=3)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        flat_ids = rearrange(token_ids, "b s -> (b s)")
        result = torch.index_select(self.embedding_table, 0, flat_ids)
        return rearrange(result, "(b s) d -> b s d", b=b, s=s)