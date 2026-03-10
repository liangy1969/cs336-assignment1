import torch
from torch import Tensor
from torch.nn import Module
from einops import rearrange, einsum, repeat
from jaxtyping import Bool, Float, Int
from cs336_basics.rope import RoPE


def dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None) -> Float[Tensor, " ... queries d_v"]:

    QK_matmul = einsum(Q, K, '... q d_k, ... k d_k -> ... q k')
    d_k = Q.shape[-1]
    if mask is not None:
        QK_matmul = QK_matmul.masked_fill(~mask, float('-inf'))
    QK_softmax = torch.softmax(QK_matmul / (d_k**0.5), dim=-1)
    result = einsum(QK_softmax, V, '... q k, ... k d_v -> ... q d_v')
    return result


class MultiHeadAttention(Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int,
                 enable_rope = False, 
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_per_head = d_model // num_heads
        weight_std = (1 / d_model) ** 0.5
        self.q_weights = torch.nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.q_weights, mean=0.0, std=weight_std, a=-3.0 * weight_std, b=3.0 * weight_std)
        self.k_weights = torch.nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.k_weights, mean=0.0, std=weight_std, a=-3.0 * weight_std, b=3.0 * weight_std)
        self.v_weights = torch.nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.v_weights, mean=0.0, std=weight_std, a=-3.0 * weight_std, b=3.0 * weight_std)
        self.o_weights = torch.nn.Parameter(torch.empty((d_model, d_model), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.o_weights, mean=0.0, std=weight_std, a=-3.0 * weight_std, b=3.0 * weight_std)
        self.enable_rope = enable_rope
        self.rope = None
        if self.enable_rope:
            assert max_seq_len is not None and theta is not None
            self.rope = RoPE(theta, d_k=self.d_per_head, max_seq_len=max_seq_len, device=device)


    def forward(self, 
                x: Float[Tensor, " ... seq d"], 
                token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d"]:
        Q = einsum(x, self.q_weights, '... d, o d -> ... o')
        K = einsum(x, self.k_weights, '... d, o d -> ... o')
        V = einsum(x, self.v_weights, '... d, o d -> ... o')
        Q = rearrange(Q, '... (num_heads d_per_head) -> ... num_heads d_per_head', num_heads=self.num_heads)
        K = rearrange(K, '... (num_heads d_per_head) -> ... num_heads d_per_head', num_heads=self.num_heads)
        V = rearrange(V, '... (num_heads d_per_head) -> ... num_heads d_per_head', num_heads=self.num_heads)
        mask = torch.tril(torch.ones(x.shape[:-1] + (x.shape[-2],), device=x.device, dtype=torch.bool))
        attention_list = []
        for i in range(self.num_heads):
            Q_i = Q[..., i, :]
            K_i = K[..., i, :]
            V_i = V[..., i, :]
            if self.enable_rope and self.rope is not None:
                if token_positions is None:
                    token_positions = torch.arange(x.shape[-2], device=x.device)
                    batch_dims = x.shape[:-2]  # everything except seq and d
                    axes = [f'n{i}' for i in range(len(batch_dims))]
                    sizes = {f'n{i}': s for i, s in enumerate(batch_dims)}
                    pattern = 's -> ' + ' '.join(axes) + ' s'
                    token_positions = repeat(token_positions, pattern, **sizes)
                Q_i = self.rope(Q_i, token_positions)
                K_i = self.rope(K_i, token_positions)
            attention_i = dot_product_attention(Q_i, K_i, V_i, mask=mask)
            attention_list.append(attention_i)
        attention_concat = rearrange(attention_list, 'num_heads ... d_per_head -> ... (num_heads d_per_head)')
        return einsum(attention_concat, self.o_weights, '... o, d o -> ... d')

            





