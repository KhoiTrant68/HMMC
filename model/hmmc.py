import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.models.utils import update_registered_buffers
from timm.layers import trunc_normal_

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
# PART 2: HAAR WAVELET TRANSFORMS
# ==========================================


def get_haar_filters(channels):
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    hl = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
    lh = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
    filters = torch.stack([ll, hl, lh, hh], dim=0).unsqueeze(1)
    return filters.repeat(channels, 1, 1, 1)


class DWT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.register_buffer("filters", get_haar_filters(channels))

    def forward(self, x):
        return F.conv2d(x, self.filters, stride=2, groups=x.shape[1])


class IDWT(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.register_buffer("filters", get_haar_filters(channels))

    def forward(self, x):
        return F.conv_transpose2d(x, self.filters, stride=2, groups=x.shape[1] // 4)


# ==========================================
# PART 3: MAMBA BACKBONE
# ==========================================


class MambaBlock(nn.Module):
    def __init__(
        self, dim, ssm_ratio=2.0, d_state=16, dt_rank="auto", d_conv=3, drop_path=0.0
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
        shortcut = x
        x = self.mamba(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.norm(x) + shortcut


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
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    dim=input_dim,
                    ssm_ratio=ssm_ratio,
                    d_state=d_state,
                    drop_path=drop_path,
                )
                for _ in range(num_blocks)
            ]
        )
        self.proj = (
            nn.Conv2d(input_dim, output_dim, 1)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


# ==========================================
# PART 4: ADVANCED ENTROPY BLOCKS
# ==========================================


class NAFBlock(nn.Module):
    def __init__(self, dim, inter_dim=None):
        super().__init__()
        self.dim = inter_dim if inter_dim is not None else dim
        dw_channel = self.dim * 2
        ffn_channel = self.dim * 2

        self.dwconv = nn.Sequential(
            nn.Conv2d(self.dim, dw_channel, 1),
            nn.Conv2d(dw_channel, dw_channel, 3, 1, padding=1, groups=dw_channel),
        )
        self.sca = nn.Sequential(
            nn.Conv2d(
                dw_channel // 2, dw_channel // 2, 3, 1, 1, groups=dw_channel // 2
            ),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1),
        )
        self.FFN = nn.Sequential(
            nn.Conv2d(self.dim, ffn_channel, 1),
            SimpleGate(),
            nn.Conv2d(ffn_channel // 2, self.dim, 1),
        )
        self.norm1 = LayerNorm2d(self.dim)
        self.norm2 = LayerNorm2d(self.dim)
        self.conv1 = nn.Conv2d(dw_channel // 2, self.dim, 1)

        self.beta = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, self.dim, 1, 1)), requires_grad=True)

        self.in_conv = (
            nn.Conv2d(dim, inter_dim, 1) if inter_dim is not None else nn.Identity()
        )
        self.out_conv = (
            nn.Conv2d(inter_dim, dim, 1) if inter_dim is not None else nn.Identity()
        )

    def forward(self, x):
        x_in = self.in_conv(x)
        identity = x_in
        x = self.norm1(x_in)

        x_dw = self.dwconv(x)
        x1, x2 = x_dw.chunk(2, dim=1)
        x = x1 * x2

        x = x * self.sca(x)
        x = self.conv1(x)
        out = identity + x * self.beta

        identity = out
        out = self.norm2(out)
        out = self.FFN(out)
        out = identity + out * self.gamma

        return self.out_conv(out)


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
        B, C, H, W = even.shape
        if dim == 2:
            return torch.stack((even, odd), dim=3).view(B, C, H * 2, W)
        else:
            return torch.stack((even, odd), dim=4).view(B, C, H, W * 2)

    def forward(self, ll, hf):
        C = ll.shape[1]
        hl, lh, hh = torch.split(hf, C, dim=1)
        l_horz = self._inverse_lifting(ll, hl, self.P_vert, self.U_vert, dim=2)
        h_horz = self._inverse_lifting(lh, hh, self.P_vert, self.U_vert, dim=2)
        x = self._inverse_lifting(l_horz, h_horz, self.P_horz, self.U_horz, dim=3)
        return x


class MultiScaleAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s = nn.Conv2d(dim, dim, 1)
        self.spatial_atte = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid()
        )
        self.dense = nn.Sequential(
            nn.Sequential(
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.Conv2d(dim, dim, 1),
            ),
            nn.Sequential(
                nn.GELU(),
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
                nn.Conv2d(dim, dim, 1),
            ),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
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
        mlp_rate=2,
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

        self.ln_low = nn.LayerNorm(c_block)
        self.q_low = nn.Linear(c_block, c_block, bias=qkv_bias)
        self.k_low = nn.Linear(c_block, c_block, bias=qkv_bias)
        self.ln_dict_low = nn.LayerNorm(c_block)
        self.scale = c_block**-0.5
        self.dict_low = nn.Parameter(torch.randn(64, c_block))

        self.router_ln = nn.LayerNorm(self.dim_high + c_block)
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

        self.msa = MultiScaleAggregation(c_block)
        self.ln_scale = LayerNorm2d(c_block)
        self.res_scale_1 = Scale(c_block, init_value=1.0)

        self.ln_mlp = LayerNorm2d(c_block)
        self.mlp = nn.Sequential(
            nn.Conv2d(c_block, c_block * mlp_rate, 1),
            nn.Conv2d(
                c_block * mlp_rate,
                c_block * mlp_rate,
                3,
                1,
                1,
                groups=c_block * mlp_rate,
            ),
            nn.GELU(),
            nn.Conv2d(c_block * mlp_rate, c_block, 1),
        )
        self.res_scale_2 = Scale(c_block, init_value=1.0)

        self.register_buffer("expert_biases", torch.zeros(num_experts))

        trunc_normal_(self.dict_low, std=0.02)
        trunc_normal_(self.experts_high, std=0.02)

    def process_low_freq(self, x):
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
        routing_logits = self.router(self.router_ln(torch.cat([hf, lf], dim=-1)))
        biased_logits = routing_logits + self.expert_biases.view(1, 1, 1, -1)

        topk_probs, topk_indices = torch.topk(
            F.softmax(biased_logits, dim=-1), k=2, dim=-1
        )
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        q = self.q_high(self.ln_high(hf)).view(B * H * W, C_high)

        reshaped_keys = self.k_high(self.ln_dict_high(self.experts_high)).view(
            self.num_experts, -1, C_high
        )
        reshaped_vals = self.v_all(self.experts_high).view(self.num_experts, -1, C_high)

        final_out = torch.zeros(B * H * W, C_high, device=hf.device, dtype=hf.dtype)

        # Process specifically masked pixels per expert
        for k_idx in range(2):
            expert_indices_k = topk_indices[..., k_idx].view(-1)
            probs_k = topk_probs[..., k_idx].view(-1, 1)

            for exp_id in range(self.num_experts):
                mask = expert_indices_k == exp_id
                if not mask.any():
                    continue

                q_masked = q[mask]  # [num_tokens_for_expert, C_high]
                k_exp = reshaped_keys[exp_id]  # [expert_entries, C_high]
                v_exp = reshaped_vals[exp_id]  # [expert_entries, C_high]

                attn = torch.matmul(q_masked, k_exp.transpose(0, 1)) * (C_high**-0.5)
                attn = F.softmax(attn, dim=-1)

                expert_out = torch.matmul(attn, v_exp)
                final_out[mask] += expert_out * probs_k[mask]

        return final_out.view(B, H, W, C_high), (routing_logits, topk_indices)

    def forward(self, x):
        x_emb = self.x_trans(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        ll, hf = self.dwt(x_emb)
        ll_processed = self.process_low_freq(ll)
        hf_processed, routing_data = self.process_high_freq_guided(
            hf.permute(0, 2, 3, 1), ll_processed.permute(0, 2, 3, 1)
        )
        hf_processed = hf_processed.permute(0, 3, 1, 2)
        recon = self.idwt(ll_processed, hf_processed)

        recon = recon + self.res_scale_1(self.msa(self.ln_scale(recon)))
        recon = recon + self.res_scale_2(self.mlp(self.ln_mlp(recon)))

        out = self.output_trans(recon.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out, routing_data


# ==========================================
# PART 5: CHECKERBOARD LOGIC
# ==========================================


class CheckerboardSplitter(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5)
        anchor = x_reshaped[..., 0, 0]
        non_anchor = torch.cat(
            [x_reshaped[..., 0, 1], x_reshaped[..., 1, 0], x_reshaped[..., 1, 1]], dim=1
        )
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
# PART 6: INTEGRATED MODEL (SOTA ALIGNED)
# ==========================================


class HMMC(CompressionModel):
    def __init__(self, N=128, M=256):
        super().__init__()
        self.N = N
        self.M = M

        self.groups = [0, 16, 32, 72, 136]  # Sums perfectly to M=256
        self.num_standard_slices = len(self.groups) - 2  # 3 standard slices
        self.last_slice_dim = self.groups[-1]

        feature_dim = [96, 128, 192]
        block_counts = [2, 2, 4]

        self.g_a = nn.Sequential(
            ConvBottleneckBlockWithStride(3, feature_dim[0]),
            MambaBlockSequence(
                feature_dim[0], feature_dim[0], num_blocks=block_counts[0]
            ),
            ConvBottleneckBlockWithStride(feature_dim[0], feature_dim[1]),
            MambaBlockSequence(
                feature_dim[1], feature_dim[1], num_blocks=block_counts[1]
            ),
            ConvBottleneckBlockWithStride(feature_dim[1], feature_dim[2]),
            MambaBlockSequence(
                feature_dim[2], feature_dim[2], num_blocks=block_counts[2]
            ),
            nn.Conv2d(feature_dim[2], M, kernel_size=5, stride=2, padding=2),
        )

        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(
                M, feature_dim[2], kernel_size=5, stride=2, output_padding=1, padding=2
            ),
            MambaBlockSequence(
                feature_dim[2], feature_dim[2], num_blocks=block_counts[2]
            ),
            ConvBottleneckBlockWithUpsample(feature_dim[2], feature_dim[1]),
            MambaBlockSequence(
                feature_dim[1], feature_dim[1], num_blocks=block_counts[1]
            ),
            ConvBottleneckBlockWithUpsample(feature_dim[1], feature_dim[0]),
            MambaBlockSequence(
                feature_dim[0], feature_dim[0], num_blocks=block_counts[0]
            ),
            ConvBottleneckBlockWithUpsample(feature_dim[0], 3),
        )

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

        # Context components
        self.context_bottlenecks = nn.ModuleList()
        self.dt_cross_attention = nn.ModuleList()
        self.context_transforms = nn.ModuleList()
        self.mean_transforms = nn.ModuleList()
        self.scale_transforms = nn.ModuleList()
        self.lrp_transforms = nn.ModuleList()

        bottleneck_dim = 192
        cum_channels = 0

        for i in range(self.num_standard_slices):
            current_dim = self.groups[i + 1]
            raw_query_dim = (M * 2) + cum_channels

            self.context_bottlenecks.append(nn.Conv2d(raw_query_dim, bottleneck_dim, 1))

            self.dt_cross_attention.append(
                SpectralMoEDictionaryCrossAttention(
                    input_dim=bottleneck_dim,
                    output_dim=M,
                    head_num=4,
                    mlp_rate=2,
                    num_experts=4,
                )
            )

            support_dim = M + bottleneck_dim
            self.context_transforms.append(NAFBlock(support_dim, inter_dim=128))
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

        self.prev_slices_down = nn.Conv2d(
            cum_channels, cum_channels, kernel_size=2, stride=2
        )
        self.hyper_down = nn.Conv2d(M * 2, M * 2, kernel_size=2, stride=2)

        self.checkerboard_split = CheckerboardSplitter()
        self.checkerboard_merge = CheckerboardMerger()

        # Anchor Context
        raw_query_dim_anc = (M * 2) + cum_channels
        self.anc_bottleneck = nn.Conv2d(raw_query_dim_anc, bottleneck_dim, 1)
        self.moe_anchor = SpectralMoEDictionaryCrossAttention(
            input_dim=bottleneck_dim,
            output_dim=M,
            head_num=4,
            mlp_rate=2,
            num_experts=4,
        )
        support_dim_anc = M + bottleneck_dim
        self.naf_anchor = NAFBlock(support_dim_anc, inter_dim=128)
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

        # Non-Anchor Context
        raw_query_dim_na = (M * 2) + cum_channels + self.last_slice_dim
        self.na_bottleneck = nn.Conv2d(raw_query_dim_na, bottleneck_dim, 1)
        self.moe_non_anchor = SpectralMoEDictionaryCrossAttention(
            input_dim=bottleneck_dim,
            output_dim=M,
            head_num=4,
            mlp_rate=2,
            num_experts=4,
        )
        support_dim_na = M + bottleneck_dim
        self.naf_non_anchor = NAFBlock(support_dim_na, inter_dim=128)
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
        y_hat_slices, y_likelihood, all_logits = [], [], []

        for i in range(self.num_standard_slices):
            y_slice = y_slices[i]
            query_raw = (
                hyper_info
                if i == 0
                else torch.cat([hyper_info, torch.cat(y_hat_slices, dim=1)], dim=1)
            )
            query = self.context_bottlenecks[i](query_raw)

            dict_info, routing_data = self.dt_cross_attention[i](query)
            all_logits.append(routing_data)

            support = torch.cat([dict_info, query], dim=1)
            support_feat = self.context_transforms[i](support)
            mu = self.mean_transforms[i](support_feat)[:, :, : y_shape[0], : y_shape[1]]
            scale = self.scale_transforms[i](support_feat)[
                :, :, : y_shape[0], : y_shape[1]
            ]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            y_hat_slice = (
                y_slice + torch.empty_like(y_slice).uniform_(-0.5, 0.5)
                if self.training and training_mode == "noise"
                else ste_round(y_slice - mu) + mu
            )
            lrp = self.lrp_transforms[i](torch.cat([support_feat, y_hat_slice], dim=1))
            y_hat_slices.append(y_hat_slice + (0.5 * torch.tanh(lrp)))

        last_slice = y_slices[-1]
        y_anchor, y_non_anchor = self.checkerboard_split(last_slice)

        prev_slices_down = self.prev_slices_down(torch.cat(y_hat_slices, dim=1))
        hyper_down = self.hyper_down(hyper_info)

        query_anc_raw = torch.cat([hyper_down, prev_slices_down], dim=1)
        query_anc = self.anc_bottleneck(query_anc_raw)

        dict_info_anc, routing_data_anc = self.moe_anchor(query_anc)
        all_logits.append(routing_data_anc)

        feat_anc = self.naf_anchor(torch.cat([dict_info_anc, query_anc], dim=1))
        mu_anc, scale_anc = self.mean_anchor(feat_anc), self.scale_anchor(feat_anc)
        _, y_lik_anc = self.gaussian_conditional(y_anchor, scale_anc, mu_anc)

        y_hat_anc = (
            y_anchor + torch.empty_like(y_anchor).uniform_(-0.5, 0.5)
            if self.training and training_mode == "noise"
            else ste_round(y_anchor - mu_anc) + mu_anc
        )
        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        query_na_raw = torch.cat([query_anc_raw, y_hat_anc], dim=1)
        query_na = self.na_bottleneck(query_na_raw)

        dict_info_na, routing_data_na = self.moe_non_anchor(query_na)
        all_logits.append(routing_data_na)

        feat_na = self.naf_non_anchor(torch.cat([dict_info_na, query_na], dim=1))
        mu_na, scale_na = self.mean_non_anchor(feat_na), self.scale_non_anchor(feat_na)
        _, y_lik_na = self.gaussian_conditional(y_non_anchor, scale_na, mu_na)

        y_hat_na = (
            y_non_anchor + torch.empty_like(y_non_anchor).uniform_(-0.5, 0.5)
            if self.training and training_mode == "noise"
            else ste_round(y_non_anchor - mu_na) + mu_na
        )
        lrp_na = self.lrp_non_anchor(torch.cat([feat_na, y_hat_na], dim=1))
        y_hat_na = y_hat_na + (0.5 * torch.tanh(lrp_na))

        y_hat_last = self.checkerboard_merge(y_hat_anc, y_hat_na)
        y_lik_last = self.checkerboard_merge(y_lik_anc, y_lik_na)

        y_hat_slices.append(y_hat_last)
        y_likelihood.append(y_lik_last)

        x_hat = self.g_s(torch.cat(y_hat_slices, dim=1))

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": torch.cat(y_likelihood, dim=1), "z": z_likelihoods},
            "router_logits": tuple(all_logits),
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
        y_hat_slices, all_symbols, all_indexes = [], [], []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()

        for i in range(self.num_standard_slices):
            y_slice = y_slices[i]
            query_raw = (
                hyper_info
                if i == 0
                else torch.cat([hyper_info, torch.cat(y_hat_slices, dim=1)], dim=1)
            )
            query = self.context_bottlenecks[i](query_raw)

            dict_info, _ = self.dt_cross_attention[i](query)
            support_feat = self.context_transforms[i](
                torch.cat([dict_info, query], dim=1)
            )
            mu = self.mean_transforms[i](support_feat)[:, :, : y_shape[0], : y_shape[1]]
            scale = self.scale_transforms[i](support_feat)[
                :, :, : y_shape[0], : y_shape[1]
            ]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols = (y_q_slice + self.gaussian_conditional._offset).to(torch.int32)
            symbols = symbols.clamp(
                0, self.gaussian_conditional._quantized_cdf.size(1) - 2
            )

            all_symbols.append(symbols.reshape(-1))
            all_indexes.append(index.reshape(-1))

            lrp = self.lrp_transforms[i](torch.cat([support_feat, y_hat_slice], dim=1))
            y_hat_slices.append(y_hat_slice + (0.5 * torch.tanh(lrp)))

        last_slice = y_slices[-1]
        y_anc, y_na = self.checkerboard_split(last_slice)

        prev_slices_down = self.prev_slices_down(torch.cat(y_hat_slices, dim=1))
        hyper_down = self.hyper_down(hyper_info)

        query_anc_raw = torch.cat([hyper_down, prev_slices_down], dim=1)
        query_anc = self.anc_bottleneck(query_anc_raw)

        dict_anc, _ = self.moe_anchor(query_anc)
        feat_anc = self.naf_anchor(torch.cat([dict_anc, query_anc], dim=1))
        mu_anc, scale_anc = self.mean_anchor(feat_anc), self.scale_anchor(feat_anc)

        index_anc = self.gaussian_conditional.build_indexes(scale_anc)
        y_q_anc = self.gaussian_conditional.quantize(y_anc, "symbols", mu_anc)
        y_hat_anc = y_q_anc + mu_anc

        symbols_anc = (y_q_anc + self.gaussian_conditional._offset).to(torch.int32)
        symbols_anc = symbols_anc.clamp(
            0, self.gaussian_conditional._quantized_cdf.size(1) - 2
        )

        all_symbols.append(symbols_anc.reshape(-1))
        all_indexes.append(index_anc.reshape(-1))

        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        query_na_raw = torch.cat([query_anc_raw, y_hat_anc], dim=1)
        query_na = self.na_bottleneck(query_na_raw)

        dict_na, _ = self.moe_non_anchor(query_na)
        feat_na = self.naf_non_anchor(torch.cat([dict_na, query_na], dim=1))
        mu_na, scale_na = self.mean_non_anchor(feat_na), self.scale_non_anchor(feat_na)

        index_na = self.gaussian_conditional.build_indexes(scale_na)
        y_q_na = self.gaussian_conditional.quantize(y_na, "symbols", mu_na)
        y_hat_na = y_q_na + mu_na

        symbols_na = (y_q_na + self.gaussian_conditional._offset).to(torch.int32)
        symbols_na = symbols_na.clamp(
            0, self.gaussian_conditional._quantized_cdf.size(1) - 2
        )

        all_symbols.append(symbols_na.reshape(-1))
        all_indexes.append(index_na.reshape(-1))

        encoder.encode_with_indexes(
            torch.cat(all_symbols).tolist(),
            torch.cat(all_indexes).tolist(),
            cdf,
            cdf_lengths,
            offsets,
        )
        return {"strings": [[encoder.flush()], z_strings], "shape": z.size()[-2:]}

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

        for i in range(self.num_standard_slices):
            query_raw = (
                hyper_info
                if i == 0
                else torch.cat([hyper_info, torch.cat(y_hat_slices, dim=1)], dim=1)
            )
            query = self.context_bottlenecks[i](query_raw)

            dict_info, _ = self.dt_cross_attention[i](query)
            support_feat = self.context_transforms[i](
                torch.cat([dict_info, query], dim=1)
            )
            mu = self.mean_transforms[i](support_feat)[:, :, : y_shape[0], : y_shape[1]]
            scale = self.scale_transforms[i](support_feat)[
                :, :, : y_shape[0], : y_shape[1]
            ]

            index = self.gaussian_conditional.build_indexes(scale)
            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            rv = torch.tensor(rv, dtype=torch.float32, device=mu.device).reshape(
                1, -1, y_shape[0], y_shape[1]
            )

            y_q_slice = rv - self.gaussian_conditional._offset
            y_hat_slice = y_q_slice + mu

            lrp = self.lrp_transforms[i](torch.cat([support_feat, y_hat_slice], dim=1))
            y_hat_slices.append(y_hat_slice + (0.5 * torch.tanh(lrp)))

        prev_slices_down = self.prev_slices_down(torch.cat(y_hat_slices, dim=1))
        hyper_down = self.hyper_down(hyper_info)

        query_anc_raw = torch.cat([hyper_down, prev_slices_down], dim=1)
        query_anc = self.anc_bottleneck(query_anc_raw)

        dict_anc, _ = self.moe_anchor(query_anc)
        feat_anc = self.naf_anchor(torch.cat([dict_anc, query_anc], dim=1))
        mu_anc, scale_anc = self.mean_anchor(feat_anc), self.scale_anchor(feat_anc)

        index_anc = self.gaussian_conditional.build_indexes(scale_anc)
        rv_anc = decoder.decode_stream(
            index_anc.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        rv_anc = torch.tensor(
            rv_anc, dtype=torch.float32, device=mu_anc.device
        ).reshape(1, self.last_slice_dim, y_shape[0] // 2, y_shape[1] // 2)
        y_q_anc = rv_anc - self.gaussian_conditional._offset
        y_hat_anc = y_q_anc + mu_anc

        lrp_anc = self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1))
        y_hat_anc = y_hat_anc + (0.5 * torch.tanh(lrp_anc))

        query_na_raw = torch.cat([query_anc_raw, y_hat_anc], dim=1)
        query_na = self.na_bottleneck(query_na_raw)

        dict_na, _ = self.moe_non_anchor(query_na)
        feat_na = self.naf_non_anchor(torch.cat([dict_na, query_na], dim=1))
        mu_na, scale_na = self.mean_non_anchor(feat_na), self.scale_non_anchor(feat_na)

        index_na = self.gaussian_conditional.build_indexes(scale_na)
        rv_na = decoder.decode_stream(
            index_na.reshape(-1).tolist(), cdf, cdf_lengths, offsets
        )
        rv_na = torch.tensor(rv_na, dtype=torch.float32, device=mu_na.device).reshape(
            1, self.last_slice_dim * 3, y_shape[0] // 2, y_shape[1] // 2
        )
        y_q_na = rv_na - self.gaussian_conditional._offset
        y_hat_na = y_q_na + mu_na

        lrp_na = self.lrp_non_anchor(torch.cat([feat_na, y_hat_na], dim=1))
        y_hat_na = y_hat_na + (0.5 * torch.tanh(lrp_na))

        y_hat_slices.append(self.checkerboard_merge(y_hat_anc, y_hat_na))
        return {"x_hat": self.g_s(torch.cat(y_hat_slices, dim=1)).clamp(0, 1)}

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
            N = state_dict["h_a.0.conv_down.weight"].size(0)
            M = state_dict["g_a.6.weight"].size(0)
        except KeyError:
            N = 128
            M = 256
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net
