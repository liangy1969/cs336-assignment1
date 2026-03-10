import torch
from torch.nn import Module
from einops import rearrange, einsum

class RoPE(Module):
    def __init__(self, 
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device: torch.device | None = None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        seq_range = torch.arange(0, max_seq_len, device=device)
        dim_range = torch.arange(0, d_k // 2, device=device)
        theta_seq = theta ** (2 * dim_range / d_k)
        theta_matrix = einsum(seq_range, 1 / theta_seq, 's, d -> s d')
        sin_matrix = torch.sin(theta_matrix)
        cos_matrix = torch.cos(theta_matrix)
        self.register_buffer('rope_sin_matrix', sin_matrix, persistent=False)
        self.register_buffer('rope_cos_matrix', cos_matrix, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rope_sin_matrix = self.get_buffer('rope_sin_matrix')
        rope_cos_matrix = self.get_buffer('rope_cos_matrix')
        assert rope_sin_matrix is not None and rope_cos_matrix is not None, "RoPE buffers not found"
        token_positions_shape = token_positions.shape
        token_positions_flatten = rearrange(token_positions, '... -> (...)')
        sin_matrix_selected = torch.index_select(rope_sin_matrix, 0, token_positions_flatten)
        cos_matrix_selected = torch.index_select(rope_cos_matrix, 0, token_positions_flatten)
        axes = [f'n{i}' for i in range(len(token_positions_shape))]
        sizes = {f'n{i}': s for i, s in enumerate(token_positions_shape)}
        lhs = '(' + ' '.join(axes) + ')'
        rhs = ' '.join(axes)
        sin_matrix_selected = rearrange(sin_matrix_selected, f'{lhs} d -> {rhs} d', **sizes)
        cos_matrix_selected = rearrange(cos_matrix_selected, f'{lhs} d -> {rhs} d', **sizes)
        # reshape x to have real and imaginary parts
        x_re_im = rearrange(x, '... (d two) -> ... d two', two=2)
        x_re = x_re_im[..., 0]
        x_im = x_re_im[..., 1]
        x_re_rope = x_re * cos_matrix_selected - x_im * sin_matrix_selected
        x_im_rope = x_im * cos_matrix_selected + x_re * sin_matrix_selected
        x_rope = rearrange([x_re_rope, x_im_rope], 'two ... d -> ... (d two)')
        return x_rope
