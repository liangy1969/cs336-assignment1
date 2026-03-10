from sympy import im
import torch
from torch import Tensor
from torch.nn import Module, ModuleList
from cs336_basics.attention import MultiHeadAttention
from cs336_basics.linear import SwiGLU
from cs336_basics.norm import RMSNorm
from jaxtyping import Bool, Float, Int
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from tests.conftest import theta


class TransformerBlock(Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            d_model, num_heads, True, theta=theta, max_seq_len=max_seq_len
        )
        self.rms_norm_1 = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model, d_ff)
        self.rms_norm_2 = RMSNorm(d_model)

    def forward(
        self, x: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        attn_block = x + self.attention(self.rms_norm_1(x))
        ff_block = attn_block + self.swiglu(self.rms_norm_2(attn_block))
        return ff_block


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.transformer_block_list = ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)

    def forward(
        self, token_ids: Int[Tensor, " batch seq"]
    ) -> Float[Tensor, "batch seq vocab"]:
        x = self.token_embedding(token_ids)
        for block in self.transformer_block_list:
            x = block(x)
        x = self.output_norm(x)
        logits = self.output_linear(x)
        return logits
