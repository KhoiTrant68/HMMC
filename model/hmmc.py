import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.models.utils import update_registered_buffers

from modules.conv_module import (
    ConvBottleneckBlockWithStride,
    ConvBottleneckBlockWithUpsample,
)
from modules.VSS_module import SS2D 


# ==========================================
# PART 1: HELPER FUNCTIONS & BLOCKS
# ==========================================

def ste_round(x):
    return torch.round(x) - x.detach() + x

def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Scale(nn.Module):
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        if x.dim() == 4 and x.shape[1] == self.scale.shape[0]:
             return x * self.scale.view(1, -1, 1, 1)
        return x * self.scale

# ==========================================
# PART 2: MAMBA BACKBONE
# ==========================================

class MambaBlock(nn.Module):
    """
    Wrapper for SS2D (Mamba) to replace Swin Blocks.
    """
    def __init__(
        self,
        dim,
        ssm_ratio=2.0,
        d_state=16,
        dt_rank="auto",
        d_conv=3,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.mamba = SS2D(
            d_model=dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            dropout=drop_path,
            forward_type="v2",
        )
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        # Input: (B, C, H, W)
        shortcut = x
        # SS2D expects (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.mamba(x)
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        return x + shortcut

class MambaBlockSequence(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_blocks=2,
        ssm_ratio=2.0,
        d_state=16,
        drop_path=0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=input_dim,
                ssm_ratio=ssm_ratio,
                d_state=d_state,
                drop_path=drop_path,
            )
            for _ in range(num_blocks)
        ])
        
        self.proj = nn.Conv2d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.proj(x)
        return x

# ==========================================
# PART 3: ATTENTION BLOCKS 
# ==========================================

class WMSA(nn.Module):
    """ Window Multi-head Self-attention """
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W': return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]

    def forward(self, x):
        # x: B H W C
        if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        
        _, H, W, _ = x.shape
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = x.permute(0, 3, 1, 2) # BHWC -> BCHW
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1) # BCHW -> BHWC

        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding().to(sim.device), 'h p q -> h 1 1 p q')
        
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = F.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if pad_r > 0 or pad_b > 0:
            output = output[:, :H, :W, :]

        if self.type != 'W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path) -> None:
        super().__init__()
        self.block_1 = WMSA(input_dim, input_dim, head_dim, window_size, type='W')
        self.block_2 = WMSA(input_dim, output_dim, head_dim, window_size, type='SW')
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x is (B, C, H, W) -> permute for Transformer
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.block_1(self.ln1(x)))
        x = x + self.drop_path(self.block_2(self.ln2(x)))
        return x.permute(0, 3, 1, 2)

class SWAtten(nn.Module):
    """
    Drop-in replacement for NAFBlock.
    Combines Convolution with Swin Window Attention.
    """
    def __init__(self, input_dim, output_dim=None, head_dim=32, window_size=8, drop_path=0., inter_dim=192):
        super().__init__()
        if output_dim is None: output_dim = input_dim
        
        self.in_conv = nn.Conv2d(input_dim, inter_dim, 1)
        self.swin = SwinBlock(inter_dim, inter_dim, head_dim, window_size, drop_path)
        self.out_conv = nn.Conv2d(inter_dim, output_dim, 1)
        self.shortcut = nn.Identity() if input_dim == output_dim else nn.Conv2d(input_dim, output_dim, 1)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.in_conv(x)
        x = self.swin(x)
        x = self.out_conv(x)
        return x + res

# ==========================================
# PART 4: SPECTRAL MOE & WAVELETS
# ==========================================

class LiftingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

class LearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.P_horz = LiftingBlock(in_channels)
        self.U_horz = LiftingBlock(in_channels)
        self.P_vert = LiftingBlock(in_channels)
        self.U_vert = LiftingBlock(in_channels)

    def forward(self, x):
        even_h, odd_h = x[:, :, :, 0::2], x[:, :, :, 1::2]
        h_horz = odd_h - self.P_horz(even_h)
        l_horz = even_h + self.U_horz(h_horz)

        even_ll, odd_ll = l_horz[:, :, 0::2, :], l_horz[:, :, 1::2, :]
        h_ll = odd_ll - self.P_vert(even_ll)
        ll = even_ll + self.U_vert(h_ll)

        even_hh, odd_hh = h_horz[:, :, 0::2, :], h_horz[:, :, 1::2, :]
        h_hh = odd_hh - self.P_vert(even_hh)
        lh = even_hh + self.U_vert(h_hh)

        hf = torch.cat([h_ll, lh, h_hh], dim=1)
        return ll, hf

class InverseLearnableWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.P_horz = LiftingBlock(in_channels)
        self.U_horz = LiftingBlock(in_channels)
        self.P_vert = LiftingBlock(in_channels)
        self.U_vert = LiftingBlock(in_channels)

    def _inverse_lifting(self, low, high, P, U, dim):
        even = low - U(high)
        odd = high + P(even)
        if dim == 2:
            B, C, H, W = even.shape
            out = torch.empty(B, C, H * 2, W, device=even.device, dtype=even.dtype)
            out[:, :, 0::2, :] = even
            out[:, :, 1::2, :] = odd
        else:
            B, C, H, W = even.shape
            out = torch.empty(B, C, H, W * 2, device=even.device, dtype=even.dtype)
            out[:, :, :, 0::2] = even
            out[:, :, :, 1::2] = odd
        return out

    def forward(self, ll, hf):
        C = ll.shape[1]
        lh, hl, hh = torch.split(hf, C, dim=1)
        l_horz = self._inverse_lifting(ll, lh, self.P_vert, self.U_vert, dim=2)
        h_horz = self._inverse_lifting(hl, hh, self.P_vert, self.U_vert, dim=2)
        x = self._inverse_lifting(l_horz, h_horz, self.P_horz, self.U_horz, dim=3)
        return x

class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s = nn.Conv2d(dim, dim, 1)
        self.spatial_atte = nn.Sequential(
             nn.Conv2d(2, 1, 7, padding=3, bias=False),
             nn.Sigmoid()
        )
        self.dense = nn.Sequential(
            nn.Sequential(nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1, groups=dim), nn.Conv2d(dim, dim, 1)),
            nn.Sequential(nn.GELU(), nn.Conv2d(dim, dim, 3, 1, 1, groups=dim), nn.Conv2d(dim, dim, 1)),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        # Expects NCHW
        s = self.s(x)
        s_out = self.dense(s)
        avg_out = torch.mean(s_out, dim=1, keepdim=True)
        max_out, _ = torch.max(s_out, dim=1, keepdim=True)
        s_attn = self.spatial_atte(torch.cat([avg_out, max_out], dim=1))
        return s_out * s_attn

class SpectralMoEDictionaryCrossAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        mlp_rate=4,
        head_num=4,
        qkv_bias=True,
        num_experts=4,
        expert_entries=64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts

        self.dim_low = 32 * head_num
        self.dim_high = 32 * head_num * 3
        c_block = 32 * head_num

        self.dwt = LearnableWaveletTransform(c_block)
        self.idwt = InverseLearnableWaveletTransform(c_block)

        self.x_trans = nn.Linear(input_dim, c_block, bias=qkv_bias)
        self.output_trans = nn.Linear(c_block, output_dim, bias=qkv_bias)

        # Low Freq
        self.ln_low = nn.LayerNorm(c_block)
        self.q_low = nn.Linear(c_block, c_block, bias=qkv_bias)
        self.k_low = nn.Linear(c_block, c_block, bias=qkv_bias)
        self.ln_dict_low = nn.LayerNorm(c_block)
        self.scale = c_block**-0.5
        self.dict_low = nn.Parameter(torch.randn(64, c_block))

        # High Freq
        self.router = nn.Sequential(
            nn.Linear(self.dim_high + c_block, self.dim_high),
            nn.GELU(),
            nn.Linear(self.dim_high, self.dim_high // 4),
            nn.GELU(),
            nn.Linear(self.dim_high // 4, num_experts),
        )
        self.experts_high = nn.Parameter(
            torch.randn(num_experts * expert_entries, self.dim_high)
        )
        self.ln_dict_high = nn.LayerNorm(self.dim_high)
        self.ln_high = nn.LayerNorm(self.dim_high)
        self.q_high = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)
        self.k_high = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)
        self.v_all = nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias)

        # Utils
        self.msa = MultiScaleAggregation(c_block)
        self.ln_scale = LayerNorm2d(c_block) 
        self.res_scale_1 = Scale(c_block, init_value=1.0)
        
        self.ln_mlp = LayerNorm2d(c_block)
        self.mlp = nn.Sequential(
            nn.Conv2d(c_block, c_block * mlp_rate, 1), 
            nn.Conv2d(c_block*mlp_rate, c_block*mlp_rate, 3, 1, 1, groups=c_block*mlp_rate),
            nn.GELU(),
            nn.Conv2d(c_block*mlp_rate, c_block, 1)
        )
        self.res_scale_2 = Scale(c_block, init_value=1.0)

        self.register_buffer("expert_biases", torch.zeros(num_experts))
        self.last_routing_logits = None
        self.last_routing_indices = None

        trunc_normal_(self.dict_low, std=0.02)
        trunc_normal_(self.experts_high, std=0.02)

    def process_low_freq(self, x):
        # x: NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)
        x_norm = self.ln_low(x)
        q = self.q_low(x_norm)
        k = self.k_low(self.ln_dict_low(self.dict_low))
        attn = torch.matmul(q, k.transpose(0, 1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, self.dict_low)
        return out.permute(0, 3, 1, 2)

    def process_high_freq_guided(self, hf, lf):
        B, H, W, C_high = hf.shape
        routing_logits = self.router(torch.cat([hf, lf], dim=-1))
        self.last_routing_logits = routing_logits
        biased_logits = routing_logits + self.expert_biases.view(1, 1, 1, -1)
        _, topk_indices = torch.topk(biased_logits, k=2, dim=-1)
        self.last_routing_indices = topk_indices

        unbiased_probs = F.softmax(routing_logits, dim=-1)
        mask = torch.zeros_like(unbiased_probs).scatter_(-1, topk_indices, 1.0)
        masked_probs = unbiased_probs * mask
        routing_weights = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

        q = self.q_high(self.ln_high(hf)).reshape(-1, C_high)
        all_keys = self.k_high(self.ln_dict_high(self.experts_high))

        sim = torch.matmul(q, all_keys.transpose(0, 1)) * (C_high**-0.5)
        sim = sim.view(B, H * W, self.num_experts, -1)
        attn = F.softmax(sim, dim=-1)

        v_experts = self.v_all(self.experts_high).view(self.num_experts, -1, C_high)
        expert_outputs = torch.einsum("bhke,kec->bhkc", attn, v_experts)
        router_weights_flat = routing_weights.view(B, H * W, self.num_experts, 1)
        final_out = (expert_outputs * router_weights_flat).sum(dim=2)

        return final_out.view(B, H, W, C_high)

    def forward(self, x):
        shortcut = x
        x_emb = self.x_trans(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        ll, hf = self.dwt(x_emb)
        ll_processed = self.process_low_freq(ll)
        hf_processed = self.process_high_freq_guided(
            hf.permute(0, 2, 3, 1), ll_processed.permute(0, 2, 3, 1)
        )
        hf_processed = hf_processed.permute(0, 3, 1, 2)
        recon = self.idwt(ll_processed, hf_processed)

        # Utils (Flow is NCHW)
        recon = recon + self.res_scale_1(self.msa(self.ln_scale(recon)))
        recon = recon + self.res_scale_2(self.mlp(self.ln_mlp(recon))) 
        
        # Output Linear expects NHWC
        out = self.output_trans(recon.permute(0, 2, 3, 1)) 
        out = out.permute(0, 3, 1, 2)
        
        if self.input_dim == self.output_dim:
            out = out + shortcut
        return out

class CheckerboardSplitter(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5)
        anchor = x_reshaped[..., 0, 0]
        non_anchor = torch.cat([x_reshaped[..., 0, 1], x_reshaped[..., 1, 0], x_reshaped[..., 1, 1]], dim=1)
        return anchor, non_anchor

class CheckerboardMerger(nn.Module):
    def forward(self, anchor, non_anchor):
        B, C, H_half, W_half = anchor.shape
        na1, na2, na3 = torch.split(non_anchor, C, dim=1)
        row0 = torch.stack([anchor, na1], dim=-1)
        row1 = torch.stack([na2, na3], dim=-1)
        grid = torch.stack([row0, row1], dim=-2)
        x = grid.permute(0, 1, 2, 4, 3, 5)
        return x.reshape(B, C, H_half * 2, W_half * 2)

# ==========================================
# PART 5: INTEGRATED MODEL 
# ==========================================

class HMMC(CompressionModel):
    def __init__(
        self,
        N=192,
        M=320,
    ):
        super().__init__()
        self.N = N
        self.M = M

        self.groups = [0, 16, 16, 32, 64, 192]
        self.num_standard_slices = len(self.groups) - 2
        self.last_slice_dim = self.groups[-1]

        # BACKBONE: MAMBA
        feature_dim = [96, 144, 256]
        block_counts = [2, 3, 6] 

        # Encoder
        self.g_a = nn.Sequential(
            ConvBottleneckBlockWithStride(3, feature_dim[0]),
            MambaBlockSequence(feature_dim[0], feature_dim[0], num_blocks=block_counts[0]),
            ConvBottleneckBlockWithStride(feature_dim[0], feature_dim[1]),
            MambaBlockSequence(feature_dim[1], feature_dim[1], num_blocks=block_counts[1]),
            ConvBottleneckBlockWithStride(feature_dim[1], feature_dim[2]),
            MambaBlockSequence(feature_dim[2], feature_dim[2], num_blocks=block_counts[2]),
            nn.Conv2d(feature_dim[2], M, kernel_size=5, stride=2, padding=2),
        )

        # Decoder
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(
                M, feature_dim[2], kernel_size=5, stride=2, output_padding=1, padding=2
            ),
            MambaBlockSequence(feature_dim[2], feature_dim[2], num_blocks=block_counts[2]),
            ConvBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1]),
            MambaBlockSequence(feature_dim[1], feature_dim[1], num_blocks=block_counts[1]),
            ConvBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0]),
            MambaBlockSequence(feature_dim[0], feature_dim[0], num_blocks=block_counts[0]),
            ConvBottleneckBlockWithUpsample(feature_dim[0], 3),
        )

        # Hyper-Prior
        self.h_a = nn.Sequential(
            ConvBottleneckBlockWithStride(M, N),
            MambaBlock(N),
            nn.Conv2d(N, 192, kernel_size=3, stride=2, padding=1),
        )
        self.h_z_s1 = nn.Sequential(
            nn.ConvTranspose2d(
                192, N, kernel_size=3, stride=2, output_padding=1, padding=1
            ),
            MambaBlock(N),
            ConvBottleneckBlockWithUpsample(N, M),
        )
        self.h_z_s2 = nn.Sequential(
            nn.ConvTranspose2d(
                192, N, kernel_size=3, stride=2, output_padding=1, padding=1
            ),
            MambaBlock(N),
            ConvBottleneckBlockWithUpsample(N, M),
        )

        # ENTROPY
        self.dt_cross_attention = nn.ModuleList()
        self.context_transforms = nn.ModuleList()
        self.mean_transforms = nn.ModuleList()
        self.scale_transforms = nn.ModuleList()
        self.lrp_transforms = nn.ModuleList()

        cum_channels = 0

        # A. Standard Slices
        for i in range(self.num_standard_slices):
            current_dim = self.groups[i + 1]
            moe_input_dim = (M * 2) + cum_channels

            self.dt_cross_attention.append(
                SpectralMoEDictionaryCrossAttention(
                    input_dim=moe_input_dim,
                    output_dim=M,
                    head_num=8,
                    mlp_rate=4,
                    num_experts=4,
                )
            )
            support_dim = M + (M * 2) + cum_channels
            
            self.context_transforms.append(
                SWAtten(input_dim=support_dim, output_dim=support_dim, 
                        inter_dim=192, window_size=8)
            )

            self.mean_transforms.append(
                nn.Sequential(
                    nn.Conv2d(support_dim, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )
            self.scale_transforms.append(
                nn.Sequential(
                    nn.Conv2d(support_dim, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )
            self.lrp_transforms.append(
                nn.Sequential(
                    nn.Conv2d(support_dim + current_dim, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, current_dim, 3, 1, 1),
                )
            )
            cum_channels += current_dim

        # B. Checkerboard
        self.checkerboard_split = CheckerboardSplitter()
        self.checkerboard_merge = CheckerboardMerger()

        self.moe_anchor = SpectralMoEDictionaryCrossAttention(
            input_dim=(M * 2) + cum_channels,
            output_dim=M,
            head_num=8,
            mlp_rate=4,
            num_experts=4,
        )
        support_dim_anc = M + (M * 2) + cum_channels
        
        # --- Mamba Scan for Anchor Context ---
        self.anchor_mamba = MambaBlock(support_dim_anc, d_state=16)
        
        # --- SWAtten for Anchor ---
        self.naf_anchor = SWAtten(input_dim=support_dim_anc, output_dim=support_dim_anc, 
                                  inter_dim=192, window_size=8)

        self.mean_anchor = nn.Sequential(
            nn.Conv2d(support_dim_anc, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim, 3, 1, 1),
        )
        self.scale_anchor = nn.Sequential(
            nn.Conv2d(support_dim_anc, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim, 3, 1, 1),
        )
        self.lrp_anchor = nn.Sequential(
            nn.Conv2d(support_dim_anc + self.last_slice_dim, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim, 3, 1, 1),
        )

        fusion_input_dim = (M * 2) + cum_channels + self.last_slice_dim
        self.moe_non_anchor = SpectralMoEDictionaryCrossAttention(
            input_dim=fusion_input_dim,
            output_dim=M,
            head_num=8,
            mlp_rate=4,
            num_experts=4,
        )
        support_dim_na = M + fusion_input_dim
        
        # --- Mamba Scan for Non-Anchor Context ---
        self.non_anchor_mamba = MambaBlock(support_dim_na, d_state=16)

        # --- SWAtten for Non-Anchor ---
        self.naf_non_anchor = SWAtten(input_dim=support_dim_na, output_dim=support_dim_na, 
                                      inter_dim=192, window_size=8)
        
        out_na_dim = self.last_slice_dim * 3

        self.mean_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, out_na_dim, 3, 1, 1),
        )
        self.scale_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, out_na_dim, 3, 1, 1),
        )
        self.lrp_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na + out_na_dim, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, out_na_dim, 3, 1, 1),
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, training_mode="noise"):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        hyper_info = torch.cat([latent_means, latent_scales], dim=1)

        y_slices = y.split(self.groups[1:], 1)
        y_hat_slices = []
        y_likelihood = []
        all_logits = []

        # A. Standard Slices
        for i in range(self.num_standard_slices):
            y_slice = y_slices[i]
            if i == 0:
                query = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([hyper_info, prev_slices], dim=1)

            dict_info = self.dt_cross_attention[i](query)
            if hasattr(self.dt_cross_attention[i], "last_routing_logits"):
                all_logits.append((
                    self.dt_cross_attention[i].last_routing_logits,
                    self.dt_cross_attention[i].last_routing_indices,
                ))

            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)
            mu = self.mean_transforms[i](support_feat)
            scale = self.scale_transforms[i](support_feat)

            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            if self.training and training_mode == "noise":
                noise = torch.empty_like(y_slice).uniform_(-0.5, 0.5)
                y_hat_slice = y_slice + noise
            else:
                y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            
            y_hat_slices.append(y_hat_slice)

        # B. Checkerboard
        last_slice = y_slices[-1]
        y_anchor, y_non_anchor = self.checkerboard_split(last_slice)

        prev_slices_full = torch.cat(y_hat_slices, dim=1)
        prev_slices_down = F.avg_pool2d(prev_slices_full, 2)
        hyper_down = F.avg_pool2d(hyper_info, 2)

        # Anchor
        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        dict_info_anc = self.moe_anchor(query_anc)
        if hasattr(self.moe_anchor, "last_routing_logits"):
            all_logits.append((
                self.moe_anchor.last_routing_logits,
                self.moe_anchor.last_routing_indices,
            ))
        
        support_anc = torch.cat([dict_info_anc, query_anc], dim=1)
        
        # --- Apply Mamba Context Scan ---
        support_anc = self.anchor_mamba(support_anc)
        feat_anc = self.naf_anchor(support_anc)
        mu_anc = self.mean_anchor(feat_anc)
        scale_anc = self.scale_anchor(feat_anc)
        _, y_lik_anc = self.gaussian_conditional(y_anchor, scale_anc, mu_anc)

        if self.training and training_mode == "noise":
            y_hat_anc = y_anchor + torch.empty_like(y_anchor).uniform_(-0.5, 0.5)
        else:
            y_hat_anc = ste_round(y_anchor - mu_anc) + mu_anc

        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        # Non-Anchor
        query_na = torch.cat([query_anc, y_hat_anc], dim=1)
        dict_info_na = self.moe_non_anchor(query_na)
        if hasattr(self.moe_non_anchor, "last_routing_logits"):
            all_logits.append((
                self.moe_non_anchor.last_routing_logits,
                self.moe_non_anchor.last_routing_indices,
            ))

        support_na = torch.cat([dict_info_na, query_na], dim=1)
        
        # --- Mamba Context Scan ---
        support_na = self.non_anchor_mamba(support_na)

        feat_na = self.naf_non_anchor(support_na) 
        mu_na = self.mean_non_anchor(feat_na)
        scale_na = self.scale_non_anchor(feat_na)
        _, y_lik_na = self.gaussian_conditional(y_non_anchor, scale_na, mu_na)

        if self.training and training_mode == "noise":
            y_hat_na = y_non_anchor + torch.empty_like(y_non_anchor).uniform_(-0.5, 0.5)
        else:
            y_hat_na = ste_round(y_non_anchor - mu_na) + mu_na

        lrp_na = self.lrp_non_anchor(torch.cat([feat_na, y_hat_na], dim=1))
        y_hat_na = y_hat_na + (0.5 * torch.tanh(lrp_na))

        # Merge
        y_hat_last = self.checkerboard_merge(y_hat_anc, y_hat_na)
        y_lik_last = self.checkerboard_merge(y_lik_anc, y_lik_na)
        
        y_hat_slices.append(y_hat_last)
        y_likelihood.append(y_lik_last)

        # Recon
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "router_logits": tuple(all_logits) if all_logits else None,
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        hyper_info = torch.cat([latent_means, latent_scales], dim=1)

        y_slices = y.split(self.groups[1:], 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        all_symbols = []
        all_indexes = []

        # A. Standard
        for i in range(self.num_standard_slices):
            y_slice = y_slices[i]
            if i == 0:
                query = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([hyper_info, prev_slices], dim=1)

            dict_info = self.dt_cross_attention[i](query)
            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)
            mu = self.mean_transforms[i](support_feat)
            scale = self.scale_transforms[i](support_feat)

            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = scale[:, :, : y_shape[0], : y_shape[1]]
            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            all_symbols.append(y_q_slice.reshape(-1))
            all_indexes.append(index.reshape(-1))

            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        # B. Checkerboard
        last_slice = y_slices[-1]
        y_anc, y_na = self.checkerboard_split(last_slice)
        prev_slices_down = F.avg_pool2d(torch.cat(y_hat_slices, dim=1), 2)
        hyper_down = F.avg_pool2d(hyper_info, 2)

        # Anchor
        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        dict_anc = self.moe_anchor(query_anc)
        support_anc = torch.cat([dict_anc, query_anc], dim=1)
        
        # --- Mamba Scan ---
        support_anc = self.anchor_mamba(support_anc)
        
        feat_anc = self.naf_anchor(support_anc)
        mu_anc = self.mean_anchor(feat_anc)
        scale_anc = self.scale_anchor(feat_anc)

        index_anc = self.gaussian_conditional.build_indexes(scale_anc)
        y_q_anc = self.gaussian_conditional.quantize(y_anc, "symbols", mu_anc)
        y_hat_anc = y_q_anc + mu_anc
        all_symbols.append(y_q_anc.reshape(-1))
        all_indexes.append(index_anc.reshape(-1))

        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        # Non-Anchor
        query_na = torch.cat([query_anc, y_hat_anc], dim=1)
        dict_na = self.moe_non_anchor(query_na)
        support_na = torch.cat([dict_na, query_na], dim=1)
        
        # --- Mamba Scan ---
        support_na = self.non_anchor_mamba(support_na)
        
        feat_na = self.naf_non_anchor(support_na)
        mu_na = self.mean_non_anchor(feat_na)
        scale_na = self.scale_non_anchor(feat_na)

        index_na = self.gaussian_conditional.build_indexes(scale_na)
        y_q_na = self.gaussian_conditional.quantize(y_na, "symbols", mu_na)
        all_symbols.append(y_q_na.reshape(-1))
        all_indexes.append(index_na.reshape(-1))

        encoder.encode_with_indexes(
            torch.cat(all_symbols).tolist(),
            torch.cat(all_indexes).tolist(),
            cdf,
            cdf_lengths,
            offsets,
        )
        y_string = encoder.flush()
        return {"strings": [[y_string], z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        hyper_info = torch.cat([latent_means, latent_scales], dim=1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])
        y_hat_slices = []

        # A. Standard
        for i in range(self.num_standard_slices):
            if i == 0:
                query = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                query = torch.cat([hyper_info, prev_slices], dim=1)

            dict_info = self.dt_cross_attention[i](query)
            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)
            mu = self.mean_transforms[i](support_feat)
            scale = self.scale_transforms[i](support_feat)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = scale[:, :, : y_shape[0], : y_shape[1]]
            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.tensor(rv, dtype=torch.float32, device=mu.device).reshape(
                1, -1, y_shape[0], y_shape[1]
            )
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_in = torch.cat([support_feat, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[i](lrp_in)
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        # B. Checkerboard
        prev_slices_down = F.avg_pool2d(torch.cat(y_hat_slices, dim=1), 2)
        hyper_down = F.avg_pool2d(hyper_info, 2)

        # Anchor
        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        dict_anc = self.moe_anchor(query_anc)
        support_anc = torch.cat([dict_anc, query_anc], dim=1)
        
        # --- Mamba Scan ---
        support_anc = self.anchor_mamba(support_anc)
        
        feat_anc = self.naf_anchor(support_anc)
        mu_anc = self.mean_anchor(feat_anc)
        scale_anc = self.scale_anchor(feat_anc)

        index_anc = self.gaussian_conditional.build_indexes(scale_anc)
        rv_anc = decoder.decode_stream(
            index_anc.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        rv_anc = (
            torch.tensor(rv_anc, dtype=torch.float32, device=mu_anc.device)
            .reshape(1, self.last_slice_dim, y_shape[0] // 2, y_shape[1] // 2)
        )
        y_hat_anc = self.gaussian_conditional.dequantize(rv_anc, mu_anc)
        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        # Non-Anchor
        query_na = torch.cat([query_anc, y_hat_anc], dim=1)
        dict_na = self.moe_non_anchor(query_na)
        support_na = torch.cat([dict_na, query_na], dim=1)
        
        # --- Mamba Scan ---
        support_na = self.non_anchor_mamba(support_na)
        feat_na = self.naf_non_anchor(support_na)
        mu_na = self.mean_non_anchor(feat_na)
        scale_na = self.scale_non_anchor(feat_na)

        index_na = self.gaussian_conditional.build_indexes(scale_na)
        rv_na = decoder.decode_stream(
            index_na.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        rv_na = (
            torch.tensor(rv_na, dtype=torch.float32, device=mu_na.device)
            .reshape(1, self.last_slice_dim * 3, y_shape[0] // 2, y_shape[1] // 2)
        )
        y_hat_na = self.gaussian_conditional.dequantize(rv_na, mu_na)
        lrp_na = self.lrp_non_anchor(torch.cat([feat_na, y_hat_na], dim=1))
        y_hat_na = y_hat_na + (0.5 * torch.tanh(lrp_na))

        # Merge
        y_hat_last = self.checkerboard_merge(y_hat_anc, y_hat_na)
        y_hat_slices.append(y_hat_last)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp(0, 1)
        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        try:
            N = state_dict["g_a.0.weight"].size(0)
            M = state_dict["g_a.6.weight"].size(0)
        except KeyError:
            N = 192
            M = 320
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net