import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.models.utils import update_registered_buffers

# Assumed external modules
from modules.conv_module import (
    ConvNeXtBlock,
    ConvBottleneckBlockWithStride,
    ConvBottleneckBlockWithUpsample,
)
from modules.VSS_module import SS2D 


# ==========================================
# PART 1: HELPER FUNCTIONS & WAVELET (DWT)
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

class HaarDWT(nn.Module):
    """Parameter-free Haar Wavelet Transform for spatial dimensions."""
    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) / 2.0
        HL = (x00 - x01 + x10 - x11) / 2.0
        LH = (x00 + x01 - x10 - x11) / 2.0
        HH = (x00 - x01 - x10 + x11) / 2.0

        # LL represents structural base. HF represents high-freq details.
        HF = torch.cat([HL, LH, HH], dim=1)
        return LL, HF

class HaarIDWT(nn.Module):
    """Inverse Haar Wavelet Transform."""
    def forward(self, LL, HF):
        B, C, H_half, W_half = LL.shape
        HL, LH, HH = torch.chunk(HF, 3, dim=1)

        x00 = (LL + HL + LH + HH) / 2.0
        x01 = (LL - HL + LH - HH) / 2.0
        x10 = (LL + HL - LH - HH) / 2.0
        x11 = (LL - HL - LH + HH) / 2.0

        out = torch.empty(B, C, H_half * 2, W_half * 2, device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out

# ==========================================
# PART 2: MAMBA BACKBONE & STREAMLINED CONTEXT
# ==========================================

class MambaBlock(nn.Module):
    """Wrapper for SS2D (Mamba)."""
    def __init__(self, dim, ssm_ratio=2.0, d_state=16, dt_rank="auto", d_conv=3, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.mamba = SS2D(
            d_model=dim, d_state=d_state, ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank, d_conv=d_conv, dropout=drop_path, forward_type="v2"
        )
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        shortcut = x
        x = x.permute(0, 2, 3, 1) # SS2D expects (B, H, W, C)
        x = self.mamba(x)
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        return x + shortcut

class MambaBlockSequence(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=2, ssm_ratio=2.0, d_state=16, drop_path=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaBlock(dim=input_dim, ssm_ratio=ssm_ratio, d_state=d_state, drop_path=drop_path)
            for _ in range(num_blocks)
        ])
        self.proj = nn.Conv2d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.proj(x)

class StreamlinedContextBlock(nn.Module):
    """Replaces the heavy NAFBlock for the Entropy Model."""
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1, groups=out_dim), # Highly efficient Depthwise
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# PART 3: TRUE SPARSE SPATIAL MoE
# ==========================================

class SpatialSparseMoE(nn.Module):
    """
    Routes spatial patches to specific experts. 
    Expert 0 is a lightweight conv for flat regions.
    Experts 1..K are heavy Mamba blocks for complex textures.
    """
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # Spatial Router
        self.router = nn.Conv2d(dim, num_experts, kernel_size=3, padding=1)
        
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i == 0:
                # Identity / Lightweight Expert for low-energy HF patches (sky, walls)
                self.experts.append(nn.Sequential(
                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, 1)
                ))
            else:
                # Heavy Experts for detail, textures, boundaries
                self.experts.append(MambaBlock(dim))
                
        # Buffer for Loss-Free Balancer
        self.register_buffer("expert_biases", torch.zeros(num_experts))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Route
        router_logits = self.router(x) # [B, Experts, H, W]
        biased_logits = router_logits + self.expert_biases.view(1, -1, 1, 1)
        
        routing_probs = F.softmax(biased_logits, dim=1)
        max_idx = routing_probs.argmax(dim=1, keepdim=True)
        
        # Straight-through estimator for hard routing
        hard_routing = torch.zeros_like(routing_probs).scatter_(1, max_idx, 1.0)
        routing_weights = hard_routing - routing_probs.detach() + routing_probs
        
        # 2. Sparse Execution via Spatial Masking (Preserves 2D Inductive Bias for Mamba)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = routing_weights[:, i:i+1, :, :]
            expert_in = x * mask
            expert_out = expert(expert_in)
            out = out + expert_out * mask
            
        # Format metrics for the balancer
        router_data = (
            router_logits.permute(0, 2, 3, 1).reshape(B, H*W, self.num_experts), 
            max_idx.permute(0, 2, 3, 1).reshape(B, H*W, 1)
        )
        return out, router_data

# ==========================================
# PART 4: FREQUENCY-DISENTANGLED ENCODER / DECODER
# ==========================================

class FrequencySplitEncoder(nn.Module):
    def __init__(self, M=320):
        super().__init__()
        self.dwt = HaarDWT()
        
        # LL Path (Cheap)
        self.ll_proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64)
        )
        
        # HF Path (Sparse MoE)
        self.hf_proj = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=5, stride=2, padding=2),
            SpatialSparseMoE(dim=64, num_experts=4)
        )
        
        # Fused downsampling to latent shape
        self.merge = nn.Sequential(
            nn.Conv2d(128, 192, 1),
            ConvBottleneckBlockWithStride(192, 256),
            MambaBlockSequence(256, 256, num_blocks=3),
            nn.Conv2d(256, M, kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x):
        LL, HF = self.dwt(x)
        
        ll_feat = self.ll_proj(LL)
        hf_feat, router_data = self.hf_proj[1](self.hf_proj[0](HF)) # Access MoE directly for stats
        
        fused = torch.cat([ll_feat, hf_feat], dim=1)
        y = self.merge(fused)
        
        return y, router_data

class FrequencySplitDecoder(nn.Module):
    def __init__(self, M=320):
        super().__init__()
        self.split = nn.Sequential(
            nn.ConvTranspose2d(M, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            MambaBlockSequence(256, 256, num_blocks=3),
            ConvBottleneckBlockWithUpsample(256, 192),
            nn.Conv2d(192, 128, 1)
        )
        
        self.ll_recon = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        self.hf_recon = nn.Sequential(
            SpatialSparseMoE(dim=64, num_experts=4),
            nn.ConvTranspose2d(64, 9, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        self.idwt = HaarIDWT()

    def forward(self, y):
        feat = self.split(y)
        ll_feat, hf_feat = torch.split(feat, 64, dim=1)
        
        LL_hat = self.ll_recon(ll_feat)
        
        hf_moe_out, router_data = self.hf_recon[0](hf_feat)
        HF_hat = self.hf_recon[1](hf_moe_out)
        
        x_hat = self.idwt(LL_hat, HF_hat)
        return x_hat, router_data

# ==========================================
# PART 5: CHECKERBOARD LOGIC
# ==========================================

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
# PART 6: INTEGRATED HMMC MODEL
# ==========================================

class HMMC(CompressionModel):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.N = N
        self.M = M

        # Core Frequency-Disentangled Architecture
        self.g_a = FrequencySplitEncoder(M=M)
        self.g_s = FrequencySplitDecoder(M=M)

        # Hyper-Prior Model (Standard)
        self.h_a = nn.Sequential(
            ConvBottleneckBlockWithStride(M, N),
            MambaBlock(N),
            nn.Conv2d(N, 192, kernel_size=3, stride=2, padding=1),
        )
        self.h_z_s1 = nn.Sequential(
            nn.ConvTranspose2d(192, N, kernel_size=3, stride=2, output_padding=1, padding=1),
            MambaBlock(N),
            ConvBottleneckBlockWithUpsample(N, M),
        )
        self.h_z_s2 = nn.Sequential(
            nn.ConvTranspose2d(192, N, kernel_size=3, stride=2, output_padding=1, padding=1),
            MambaBlock(N),
            ConvBottleneckBlockWithUpsample(N, M),
        )

        # Slice configuration
        self.groups = [0, 16, 16, 32, 64, 192]
        self.num_standard_slices = len(self.groups) - 2
        self.last_slice_dim = self.groups[-1]

        # ENTROPY MODEL (The Diet Version)
        self.context_transforms = nn.ModuleList()
        self.mean_transforms = nn.ModuleList()
        self.scale_transforms = nn.ModuleList()
        self.lrp_transforms = nn.ModuleList()

        cum_channels = 0
        
        # A. Standard Slices
        for i in range(self.num_standard_slices):
            current_dim = self.groups[i + 1]
            # query features = hyper_means + hyper_scales + cum_channels
            support_dim = (2 * M) + cum_channels 

            self.context_transforms.append(StreamlinedContextBlock(support_dim, 128))

            self.mean_transforms.append(nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, current_dim, 3, 1, 1),
            ))
            self.scale_transforms.append(nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, current_dim, 3, 1, 1),
            ))
            self.lrp_transforms.append(nn.Sequential(
                nn.Conv2d(128 + current_dim, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, current_dim, 3, 1, 1),
            ))
            cum_channels += current_dim

        # B. Checkerboard Components
        self.checkerboard_split = CheckerboardSplitter()
        self.checkerboard_merge = CheckerboardMerger()

        support_dim_anc = (2 * M) + cum_channels
        self.naf_anchor = StreamlinedContextBlock(support_dim_anc, 128)

        self.mean_anchor = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, self.last_slice_dim, 3, 1, 1),
        )
        self.scale_anchor = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, self.last_slice_dim, 3, 1, 1),
        )
        self.lrp_anchor = nn.Sequential(
            nn.Conv2d(128 + self.last_slice_dim, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, self.last_slice_dim, 3, 1, 1),
        )

        support_dim_na = support_dim_anc + self.last_slice_dim
        self.naf_non_anchor = StreamlinedContextBlock(support_dim_na, 128)
        out_na_dim = self.last_slice_dim * 3

        self.mean_non_anchor = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, out_na_dim, 3, 1, 1),
        )
        self.scale_non_anchor = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, out_na_dim, 3, 1, 1),
        )
        self.lrp_non_anchor = nn.Sequential(
            nn.Conv2d(128 + out_na_dim, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, out_na_dim, 3, 1, 1),
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def get_moe_modules(self):
        """Helper for train.py to easily access the MoE modules for bias updates."""
        return [self.g_a.hf_proj[1], self.g_s.hf_recon[0]]

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, training_mode="noise"):
        y, enc_router_data = self.g_a(x)
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

        # A. Standard Slices
        for i in range(self.num_standard_slices):
            y_slice = y_slices[i]
            
            # Autoregressive Query creation (No heavy cross-attention dictionary needed!)
            if i == 0:
                support = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                support = torch.cat([hyper_info, prev_slices], dim=1)

            support_feat = self.context_transforms[i](support)
            mu = self.mean_transforms[i](support_feat)
            scale = self.scale_transforms[i](support_feat)

            mu = mu[:, :, : y_shape[0], : y_shape[1]]
            scale = scale[:, :, : y_shape[0], : y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)

            if self.training and training_mode == "noise":
                y_hat_slice = y_slice + torch.empty_like(y_slice).uniform_(-0.5, 0.5)
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
        support_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
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
        support_na = torch.cat([support_anc, y_hat_anc], dim=1)
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
        
        x_hat, dec_router_data = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "router_logits": [enc_router_data, dec_router_data],
        }

    def compress(self, x):
        y, _ = self.g_a(x)
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
                support = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                support = torch.cat([hyper_info, prev_slices], dim=1)

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
        support_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
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
        support_na = torch.cat([support_anc, y_hat_anc], dim=1)
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
                support = hyper_info
            else:
                prev_slices = torch.cat(y_hat_slices, dim=1)
                support = torch.cat([hyper_info, prev_slices], dim=1)

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
        support_anc = torch.cat([hyper_down, prev_slices_down], dim=1)
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
        support_na = torch.cat([support_anc, y_hat_anc], dim=1)
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
        x_hat, _ = self.g_s(y_hat)
        
        return {"x_hat": x_hat.clamp(0, 1)}

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
        # Graceful fallback for instantiation based on params
        N, M = 192, 320
        net = cls(N=N, M=M)
        net.load_state_dict(state_dict)
        return net