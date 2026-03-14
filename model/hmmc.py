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


def ste_round(x):
    return torch.round(x) - x.detach() + x


def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


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


class MambaBlock(nn.Module):
    def __init__(
        self, dim, ssm_ratio=2.0, d_state=16, dt_rank="auto", d_conv=3, drop_path=0.0
    ):
        super().__init__()
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


class NAFBlock(nn.Module):
    def __init__(self, dim, inter_dim=None):
        super().__init__()
        self.dim = inter_dim if inter_dim is not None else dim
        dw_channel = self.dim * 2

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
            nn.Conv2d(self.dim, dw_channel, 1),
            SimpleGate(),
            nn.Conv2d(dw_channel // 2, self.dim, 1),
        )
        self.norm1 = LayerNorm2d(self.dim)
        self.norm2 = LayerNorm2d(self.dim)
        self.conv1 = nn.Conv2d(dw_channel // 2, self.dim, 1)
        self.beta = nn.Parameter(torch.zeros((1, self.dim, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, self.dim, 1, 1)))
        self.in_conv = (
            nn.Conv2d(dim, inter_dim, 1) if inter_dim is not None else nn.Identity()
        )
        self.out_conv = (
            nn.Conv2d(inter_dim, dim, 1) if inter_dim is not None else nn.Identity()
        )

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        x_dw = self.dwconv(self.norm1(x))
        x1, x2 = x_dw.chunk(2, dim=1)
        x = x1 * x2
        out = identity + self.conv1(x * self.sca(x)) * self.beta
        return self.out_conv(out + self.FFN(self.norm2(out)) * self.gamma)


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

        return ll, torch.cat([h_ll, lh, h_hh], dim=1)


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
        lh, hl, hh = torch.split(hf, C, dim=1)
        l_horz = self._inverse_lifting(ll, lh, self.P_vert, self.U_vert, dim=2)
        h_horz = self._inverse_lifting(hl, hh, self.P_vert, self.U_vert, dim=2)
        return self._inverse_lifting(l_horz, h_horz, self.P_horz, self.U_horz, dim=3)


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
        s_out = self.dense(self.s(x))
        avg_out = torch.mean(s_out, dim=1, keepdim=True)
        max_out, _ = torch.max(s_out, dim=1, keepdim=True)
        return s_out * self.spatial_atte(torch.cat([avg_out, max_out], dim=1))


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

        c_block = 32 * head_num
        self.dim_high = c_block * 3

        self.dwt = LearnableWaveletTransform(c_block)
        self.idwt = InverseLearnableWaveletTransform(c_block)

        self.x_trans = nn.Linear(input_dim, c_block, bias=qkv_bias)
        self.output_trans = nn.Linear(c_block, output_dim, bias=qkv_bias)

        self.ln_low = nn.LayerNorm(c_block)
        self.q_low, self.k_low = nn.Linear(c_block, c_block, bias=qkv_bias), nn.Linear(
            c_block, c_block, bias=qkv_bias
        )
        self.ln_dict_low = nn.LayerNorm(c_block)
        self.scale = c_block**-0.5
        self.dict_low = nn.Parameter(torch.randn(64, c_block))

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
        self.q_high, self.k_high, self.v_all = (
            nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias),
            nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias),
            nn.Linear(self.dim_high, self.dim_high, bias=qkv_bias),
        )

        self.msa = MultiScaleAggregation(c_block)
        self.ln_scale = LayerNorm2d(c_block)
        self.res_scale_1 = Scale(c_block)

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
        self.res_scale_2 = Scale(c_block)

        self.register_buffer("expert_biases", torch.zeros(num_experts))
        self.last_routing_logits, self.last_routing_indices = None, None

        trunc_normal_(self.dict_low, std=0.02)
        trunc_normal_(self.experts_high, std=0.02)

    def process_low_freq(self, x):
        x = self.ln_low(x.permute(0, 2, 3, 1))
        attn = F.softmax(
            torch.matmul(
                self.q_low(x),
                self.k_low(self.ln_dict_low(self.dict_low)).transpose(0, 1),
            )
            * self.scale,
            dim=-1,
        )
        return torch.matmul(attn, self.dict_low).permute(0, 3, 1, 2)

    def process_high_freq_guided(self, hf, lf):
        B, H, W, C_high = hf.shape
        routing_logits = self.router(torch.cat([hf, lf], dim=-1))
        self.last_routing_logits = routing_logits
        _, topk_indices = torch.topk(
            routing_logits + self.expert_biases.view(1, 1, 1, -1), k=2, dim=-1
        )
        self.last_routing_indices = topk_indices

        unbiased_probs = F.softmax(routing_logits, dim=-1)
        masked_probs = unbiased_probs * torch.zeros_like(unbiased_probs).scatter_(
            -1, topk_indices, 1.0
        )
        routing_weights = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

        sim = torch.matmul(
            self.q_high(self.ln_high(hf)).reshape(-1, C_high),
            self.k_high(self.ln_dict_high(self.experts_high)).transpose(0, 1),
        ) * (C_high**-0.5)
        sim = sim.view(B, H * W, self.num_experts, -1)

        attn = F.softmax(sim.float(), dim=-1).type_as(sim)

        expert_outputs = torch.einsum(
            "bhke,kec->bhkc",
            attn,
            self.v_all(self.experts_high).view(self.num_experts, -1, C_high),
        )
        return (
            (expert_outputs * routing_weights.view(B, H * W, self.num_experts, 1))
            .sum(dim=2)
            .view(B, H, W, C_high)
        )

    def forward(self, x):
        shortcut = x
        x_emb = self.x_trans(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        ll, hf = self.dwt(x_emb)
        ll_processed = self.process_low_freq(ll)
        recon = self.idwt(
            ll_processed,
            self.process_high_freq_guided(
                hf.permute(0, 2, 3, 1), ll_processed.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2),
        )

        recon = recon + self.res_scale_1(self.msa(self.ln_scale(recon)))
        recon = recon + self.res_scale_2(self.mlp(self.ln_mlp(recon)))

        out = self.output_trans(recon.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.input_dim == self.output_dim:
            out = out + shortcut
        return out


class CheckerboardSplitter(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 3, 5)
        return x_reshaped[..., 0, 0], torch.cat(
            [x_reshaped[..., 0, 1], x_reshaped[..., 1, 0], x_reshaped[..., 1, 1]], dim=1
        )


class CheckerboardMerger(nn.Module):
    def forward(self, anchor, non_anchor):
        B, C, H_half, W_half = anchor.shape
        na1, na2, na3 = torch.split(non_anchor, C, dim=1)
        return (
            torch.stack(
                [torch.stack([anchor, na1], dim=-1), torch.stack([na2, na3], dim=-1)],
                dim=-2,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(B, C, H_half * 2, W_half * 2)
        )


class HMMC(CompressionModel):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N, self.M = N, M
        self.groups = [0, 16, 16, 32, 64, 192]
        self.num_standard_slices, self.last_slice_dim = (
            len(self.groups) - 2,
            self.groups[-1],
        )
        feature_dim, block_counts = [96, 144, 256], [2, 3, 6]

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

        (
            self.dt_cross_attention,
            self.context_transforms,
            self.mean_transforms,
            self.scale_transforms,
            self.lrp_transforms,
        ) = (nn.ModuleList() for _ in range(5))
        cum_channels = 0

        for i in range(self.num_standard_slices):
            current_dim = self.groups[i + 1]
            self.dt_cross_attention.append(
                SpectralMoEDictionaryCrossAttention(
                    input_dim=(M * 2) + cum_channels,
                    output_dim=M,
                    head_num=8,
                    mlp_rate=4,
                    num_experts=4,
                )
            )
            support_dim = M + (M * 2) + cum_channels
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

        self.checkerboard_split, self.checkerboard_merge = (
            CheckerboardSplitter(),
            CheckerboardMerger(),
        )

        self.moe_anchor = SpectralMoEDictionaryCrossAttention(
            input_dim=(M * 2) + cum_channels,
            output_dim=M,
            head_num=8,
            mlp_rate=4,
            num_experts=4,
        )
        support_dim_anc = M + (M * 2) + cum_channels
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

        fusion_input_dim = (M * 2) + cum_channels + self.last_slice_dim
        self.moe_non_anchor = SpectralMoEDictionaryCrossAttention(
            input_dim=fusion_input_dim,
            output_dim=M,
            head_num=8,
            mlp_rate=4,
            num_experts=4,
        )
        support_dim_na = M + fusion_input_dim
        self.naf_non_anchor = NAFBlock(support_dim_na, inter_dim=128)
        self.mean_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim * 3, 3, 1, 1),
        )
        self.scale_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim * 3, 3, 1, 1),
        )
        self.lrp_non_anchor = nn.Sequential(
            nn.Conv2d(support_dim_na + self.last_slice_dim * 3, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, self.last_slice_dim * 3, 3, 1, 1),
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

        hyper_info = torch.cat([self.h_z_s2(z_hat), self.h_z_s1(z_hat)], dim=1)
        y_slices = y.split(self.groups[1:], 1)
        y_hat_slices, y_likelihood, all_logits = [], [], []

        for i in range(self.num_standard_slices):
            query = (
                hyper_info
                if i == 0
                else torch.cat([hyper_info, torch.cat(y_hat_slices, dim=1)], dim=1)
            )
            dict_info = self.dt_cross_attention[i](query)
            if hasattr(self.dt_cross_attention[i], "last_routing_logits"):
                all_logits.append(
                    (
                        self.dt_cross_attention[i].last_routing_logits,
                        self.dt_cross_attention[i].last_routing_indices,
                    )
                )

            support_feat = self.context_transforms[i](
                torch.cat([dict_info, query], dim=1)
            )
            mu = self.mean_transforms[i](support_feat)[:, :, : y_shape[0], : y_shape[1]]
            scale = self.scale_transforms[i](support_feat)[
                :, :, : y_shape[0], : y_shape[1]
            ]

            _, y_slice_likelihood = self.gaussian_conditional(y_slices[i], scale, mu)
            y_likelihood.append(y_slice_likelihood)

            y_hat_slice = (
                y_slices[i] + torch.empty_like(y_slices[i]).uniform_(-0.5, 0.5)
                if self.training and training_mode == "noise"
                else ste_round(y_slices[i] - mu) + mu
            )
            y_hat_slices.append(
                y_hat_slice
                + (
                    0.5
                    * torch.tanh(
                        self.lrp_transforms[i](
                            torch.cat([support_feat, y_hat_slice], dim=1)
                        )
                    )
                )
            )

        y_anchor, y_non_anchor = self.checkerboard_split(y_slices[-1])

        prev_slices_down = self.prev_slices_down(torch.cat(y_hat_slices, dim=1))
        hyper_down = self.hyper_down(hyper_info)

        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        dict_info_anc = self.moe_anchor(query_anc)
        if hasattr(self.moe_anchor, "last_routing_logits"):
            all_logits.append(
                (
                    self.moe_anchor.last_routing_logits,
                    self.moe_anchor.last_routing_indices,
                )
            )

        feat_anc = self.naf_anchor(torch.cat([dict_info_anc, query_anc], dim=1))
        mu_anc, scale_anc = self.mean_anchor(feat_anc), self.scale_anchor(feat_anc)
        _, y_lik_anc = self.gaussian_conditional(y_anchor, scale_anc, mu_anc)

        y_hat_anc = (
            y_anchor + torch.empty_like(y_anchor).uniform_(-0.5, 0.5)
            if self.training and training_mode == "noise"
            else ste_round(y_anchor - mu_anc) + mu_anc
        )
        y_hat_anc = y_hat_anc + (
            0.5 * torch.tanh(self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1)))
        )

        query_na = torch.cat([query_anc, y_hat_anc], dim=1)
        dict_info_na = self.moe_non_anchor(query_na)
        if hasattr(self.moe_non_anchor, "last_routing_logits"):
            all_logits.append(
                (
                    self.moe_non_anchor.last_routing_logits,
                    self.moe_non_anchor.last_routing_indices,
                )
            )

        feat_na = self.naf_non_anchor(torch.cat([dict_info_na, query_na], dim=1))
        mu_na, scale_na = self.mean_non_anchor(feat_na), self.scale_non_anchor(feat_na)
        _, y_lik_na = self.gaussian_conditional(y_non_anchor, scale_na, mu_na)

        y_hat_na = (
            y_non_anchor + torch.empty_like(y_non_anchor).uniform_(-0.5, 0.5)
            if self.training and training_mode == "noise"
            else ste_round(y_non_anchor - mu_na) + mu_na
        )
        y_hat_na = y_hat_na + (
            0.5 * torch.tanh(self.lrp_non_anchor(torch.cat([feat_na, y_hat_na], dim=1)))
        )

        y_hat_slices.append(self.checkerboard_merge(y_hat_anc, y_hat_na))
        y_likelihood.append(self.checkerboard_merge(y_lik_anc, y_lik_na))

        return {
            "x_hat": self.g_s(torch.cat(y_hat_slices, dim=1)),
            "likelihoods": {"y": torch.cat(y_likelihood, dim=1), "z": z_likelihoods},
            "router_logits": tuple(all_logits) if all_logits else None,
        }

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_info = torch.cat([self.h_z_s2(z_hat), self.h_z_s1(z_hat)], dim=1)
        y_slices, y_hat_slices = y.split(self.groups[1:], 1), []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        all_symbols, all_indexes = [], []

        for i in range(self.num_standard_slices):
            query = (
                hyper_info
                if i == 0
                else torch.cat([hyper_info, torch.cat(y_hat_slices, dim=1)], dim=1)
            )
            support_feat = self.context_transforms[i](
                torch.cat([self.dt_cross_attention[i](query), query], dim=1)
            )
            mu = self.mean_transforms[i](support_feat)[:, :, : y_shape[0], : y_shape[1]]
            scale = self.scale_transforms[i](support_feat)[
                :, :, : y_shape[0], : y_shape[1]
            ]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slices[i], "symbols", mu)
            y_hat_slice = y_q_slice + mu

            all_symbols.append(y_q_slice.reshape(-1))
            all_indexes.append(index.reshape(-1))
            y_hat_slices.append(
                y_hat_slice
                + (
                    0.5
                    * torch.tanh(
                        self.lrp_transforms[i](
                            torch.cat([support_feat, y_hat_slice], dim=1)
                        )
                    )
                )
            )

        y_anc, y_na = self.checkerboard_split(y_slices[-1])

        prev_slices_down = self.prev_slices_down(torch.cat(y_hat_slices, dim=1))
        hyper_down = self.hyper_down(hyper_info)

        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        feat_anc = self.naf_anchor(
            torch.cat([self.moe_anchor(query_anc), query_anc], dim=1)
        )
        mu_anc, scale_anc = self.mean_anchor(feat_anc), self.scale_anchor(feat_anc)

        index_anc = self.gaussian_conditional.build_indexes(scale_anc)
        y_q_anc = self.gaussian_conditional.quantize(y_anc, "symbols", mu_anc)
        y_hat_anc = y_q_anc + mu_anc
        all_symbols.append(y_q_anc.reshape(-1))
        all_indexes.append(index_anc.reshape(-1))
        y_hat_anc = y_hat_anc + (
            0.5 * torch.tanh(self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1)))
        )

        query_na = torch.cat([query_anc, y_hat_anc], dim=1)
        feat_na = self.naf_non_anchor(
            torch.cat([self.moe_non_anchor(query_na), query_na], dim=1)
        )
        mu_na, scale_na = self.mean_non_anchor(feat_na), self.scale_non_anchor(feat_na)

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
        return {"strings": [[encoder.flush()], z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        hyper_info = torch.cat([self.h_z_s2(z_hat), self.h_z_s1(z_hat)], dim=1)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])
        y_hat_slices = []

        for i in range(self.num_standard_slices):
            query = (
                hyper_info
                if i == 0
                else torch.cat([hyper_info, torch.cat(y_hat_slices, dim=1)], dim=1)
            )
            support_feat = self.context_transforms[i](
                torch.cat([self.dt_cross_attention[i](query), query], dim=1)
            )
            mu = self.mean_transforms[i](support_feat)[:, :, : y_shape[0], : y_shape[1]]
            scale = self.scale_transforms[i](support_feat)[
                :, :, : y_shape[0], : y_shape[1]
            ]
            index = self.gaussian_conditional.build_indexes(scale)

            rv = torch.tensor(
                decoder.decode_stream(
                    index.reshape(-1).tolist(), cdf, cdf_lengths, offsets
                ),
                dtype=torch.float32,
                device=mu.device,
            ).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)
            y_hat_slices.append(
                y_hat_slice
                + (
                    0.5
                    * torch.tanh(
                        self.lrp_transforms[i](
                            torch.cat([support_feat, y_hat_slice], dim=1)
                        )
                    )
                )
            )

        prev_slices_down = self.prev_slices_down(torch.cat(y_hat_slices, dim=1))
        hyper_down = self.hyper_down(hyper_info)

        query_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
        feat_anc = self.naf_anchor(
            torch.cat([self.moe_anchor(query_anc), query_anc], dim=1)
        )
        mu_anc, scale_anc = self.mean_anchor(feat_anc), self.scale_anchor(feat_anc)

        index_anc = self.gaussian_conditional.build_indexes(scale_anc)
        rv_anc = torch.tensor(
            decoder.decode_stream(
                index_anc.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            ),
            dtype=torch.float32,
            device=mu_anc.device,
        ).reshape(1, self.last_slice_dim, y_shape[0] // 2, y_shape[1] // 2)
        y_hat_anc = self.gaussian_conditional.dequantize(rv_anc, mu_anc)
        y_hat_anc = y_hat_anc + (
            0.5 * torch.tanh(self.lrp_anchor(torch.cat([feat_anc, y_hat_anc], dim=1)))
        )

        query_na = torch.cat([query_anc, y_hat_anc], dim=1)
        feat_na = self.naf_non_anchor(
            torch.cat([self.moe_non_anchor(query_na), query_na], dim=1)
        )
        mu_na, scale_na = self.mean_non_anchor(feat_na), self.scale_non_anchor(feat_na)

        index_na = self.gaussian_conditional.build_indexes(scale_na)
        rv_na = torch.tensor(
            decoder.decode_stream(
                index_na.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            ),
            dtype=torch.float32,
            device=mu_na.device,
        ).reshape(1, self.last_slice_dim * 3, y_shape[0] // 2, y_shape[1] // 2)
        y_hat_na = self.gaussian_conditional.dequantize(rv_na, mu_na)
        y_hat_na = y_hat_na + (
            0.5 * torch.tanh(self.lrp_non_anchor(torch.cat([feat_na, y_hat_na], dim=1)))
        )

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
            N, M = state_dict["g_a.0.conv_down.weight"].size(0), state_dict[
                "g_a.6.weight"
            ].size(0)
        except KeyError:
            N, M = 192, 320
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net
