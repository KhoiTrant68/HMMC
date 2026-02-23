import math
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from compressai.models.utils import update_registered_buffers

# Assumed external modules from your repository
from modules.conv_module import (
    ConvNeXtBlock,
    ConvBottleneckBlockWithStride,
    ConvBottleneckBlockWithUpsample,
)
from modules.VSS_module import SS2D 

# ==========================================
# PART 1: HELPER FUNCTIONS & IMAGE WAVELET 
# ==========================================

def ste_round(x):
    return torch.round(x) - x.detach() + x

def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ImageHaarDWT(nn.Module):
    """Parameter-free Haar Wavelet Transform for spatial dimensions (Image-Level)."""
    def forward(self, x):
        x00, x01 = x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2]
        x10, x11 = x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]
        LL = (x00 + x01 + x10 + x11) / 2.0
        HL = (x00 - x01 + x10 - x11) / 2.0
        LH = (x00 + x01 - x10 - x11) / 2.0
        HH = (x00 - x01 - x10 + x11) / 2.0
        HF = torch.cat([HL, LH, HH], dim=1)
        return LL, HF

class ImageHaarIDWT(nn.Module):
    def forward(self, LL, HF):
        B, C, H_half, W_half = LL.shape
        HL, LH, HH = torch.chunk(HF, 3, dim=1)
        x00 = (LL + HL + LH + HH) / 2.0
        x01 = (LL - HL + LH - HH) / 2.0
        x10 = (LL + HL - LH - HH) / 2.0
        x11 = (LL - HL - LH + HH) / 2.0
        out = torch.empty(B, C, H_half * 2, W_half * 2, device=LL.device, dtype=LL.dtype)
        out[:, :, 0::2, 0::2], out[:, :, 0::2, 1::2] = x00, x01
        out[:, :, 1::2, 0::2], out[:, :, 1::2, 1::2] = x10, x11
        return out

# ==========================================
# PART 2: LATENT WAVELET TRANSFORM (DCAE Inspired)
# ==========================================

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
        dim = x.shape[1]
        x_ll = F.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2).transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = F.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2).reshape(B, -1, H, W)
        C = x.shape[1]
        filters = filters.repeat(C, 1, 1, 1)
        return F.conv_transpose2d(x, filters, stride=2, groups=C)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = F.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = F.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = F.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = F.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class LatentDWT(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi, dec_lo = torch.Tensor(w.dec_hi[::-1]), torch.Tensor(w.dec_lo[::-1])
        self.register_buffer('w_ll', (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class LatentIDWT(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi, rec_lo = torch.Tensor(w.rec_hi), torch.Tensor(w.rec_lo)
        filters = torch.cat([
            (rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(1),
            (rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(1),
            (rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(1),
            (rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(1)
        ], dim=0)
        self.register_buffer('filters', filters)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

# ==========================================
# PART 3: MAMBA BACKBONE & SPATIAL MoE
# ==========================================

class MambaBlock(nn.Module):
    def __init__(self, dim, ssm_ratio=2.0, d_state=16, dt_rank="auto", drop_path=0.0):
        super().__init__()
        self.mamba = SS2D(
            d_model=dim, d_state=d_state, ssm_ratio=ssm_ratio, 
            dt_rank=dt_rank, d_conv=3, dropout=drop_path, forward_type="v2"
        )
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        shortcut = x
        x = self.mamba(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.norm(x) + shortcut

class MambaBlockSequence(nn.Module):
    def __init__(self, input_dim, output_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([MambaBlock(dim=input_dim) for _ in range(num_blocks)])
        self.proj = nn.Conv2d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        for block in self.blocks: x = block(x)
        return self.proj(x)

class SpatialSparseMoE(nn.Module):
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Conv2d(dim, num_experts, kernel_size=3, padding=1)
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if i == 0:
                self.experts.append(nn.Sequential(
                    nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU(), nn.Conv2d(dim, dim, 1)
                ))
            else:
                self.experts.append(MambaBlock(dim))
        self.register_buffer("expert_biases", torch.zeros(num_experts))

    def forward(self, x):
        B, C, H, W = x.shape
        router_logits = self.router(x)
        biased_logits = router_logits + self.expert_biases.view(1, -1, 1, 1)
        
        routing_probs = F.softmax(biased_logits, dim=1)
        max_idx = routing_probs.argmax(dim=1, keepdim=True)
        hard_routing = torch.zeros_like(routing_probs).scatter_(1, max_idx, 1.0)
        routing_weights = hard_routing - routing_probs.detach() + routing_probs
        
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = routing_weights[:, i:i+1, :, :]
            out = out + expert(x * mask) * mask
            
        router_data = (
            router_logits.permute(0, 2, 3, 1).reshape(B, H*W, self.num_experts), 
            max_idx.permute(0, 2, 3, 1).reshape(B, H*W, 1)
        )
        return out, router_data

class FrequencySplitEncoder(nn.Module):
    def __init__(self, M=320):
        super().__init__()
        self.dwt = ImageHaarDWT()
        self.ll_proj = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2), ConvNeXtBlock(64), ConvNeXtBlock(64)
        )
        self.hf_proj = nn.Sequential(
            nn.Conv2d(9, 64, 5, 2, 2), SpatialSparseMoE(dim=64, num_experts=4)
        )
        self.merge = nn.Sequential(
            nn.Conv2d(128, 192, 1), ConvBottleneckBlockWithStride(192, 256),
            MambaBlockSequence(256, 256, num_blocks=3), nn.Conv2d(256, M, 5, 2, 2)
        )

    def forward(self, x):
        LL, HF = self.dwt(x)
        hf_feat, router_data = self.hf_proj[1](self.hf_proj[0](HF))
        y = self.merge(torch.cat([self.ll_proj(LL), hf_feat], dim=1))
        return y, router_data

class FrequencySplitDecoder(nn.Module):
    def __init__(self, M=320):
        super().__init__()
        self.split = nn.Sequential(
            nn.ConvTranspose2d(M, 256, 5, 2, 2, 1), MambaBlockSequence(256, 256, num_blocks=3),
            ConvBottleneckBlockWithUpsample(256, 192), nn.Conv2d(192, 128, 1)
        )
        self.ll_recon = nn.Sequential(
            ConvNeXtBlock(64), ConvNeXtBlock(64), nn.ConvTranspose2d(64, 3, 5, 2, 2, 1)
        )
        self.hf_recon = nn.Sequential(
            SpatialSparseMoE(dim=64, num_experts=4), nn.ConvTranspose2d(64, 9, 5, 2, 2, 1)
        )
        self.idwt = ImageHaarIDWT()

    def forward(self, y):
        ll_feat, hf_feat = torch.split(self.split(y), 64, dim=1)
        hf_moe_out, router_data = self.hf_recon[0](hf_feat)
        return self.idwt(self.ll_recon(ll_feat), self.hf_recon[1](hf_moe_out)), router_data

# ==========================================
# PART 4: DCAE DICTIONARY CROSS-ATTENTION
# ==========================================

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        return self.fc2(self.act(self.dwconv(x)) * v)

class DictionaryCrossAttentionContext(nn.Module):
    def __init__(self, in_dim, out_dim=128, dict_dim=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dict_dim // num_heads) ** -0.5
        
        self.q_proj = nn.Conv2d(in_dim, dict_dim, 1)
        self.k_proj = nn.Linear(dict_dim, dict_dim)
        
        self.out_proj = nn.Conv2d(dict_dim, dict_dim, 1)
        self.glu = ConvolutionalGLU(dict_dim, dict_dim * 4, out_dim)
        self.norm_q = nn.LayerNorm(dict_dim)
        self.norm_k = nn.LayerNorm(dict_dim)
        
    def forward(self, context, global_dict):
        B, C, H, W = context.shape
        q = rearrange(self.norm_q(rearrange(self.q_proj(context), 'b c h w -> b (h w) c')), 'b n (h d) -> b h n d', h=self.num_heads)
        
        dict_norm = self.norm_k(global_dict)
        k = rearrange(self.k_proj(dict_norm), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(dict_norm, 'b t (h d) -> b h t d', h=self.num_heads)
        
        attn = torch.softmax(torch.einsum('bhnd, bhtd -> bhnt', q, k) * self.scale, dim=-1)
        dict_features = rearrange(torch.einsum('bhnt, bhtd -> bhnd', attn, v), 'b h (x y) d -> b (h d) x y', x=H, y=W)
        
        return self.glu(self.out_proj(dict_features))

# ==========================================
# PART 5: INTEGRATED HMMC ARCHITECTURE
# ==========================================

class HMMC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = num_slices
        
        # Latent DWT splits y (M) into y_low (M) and y_high (3M)
        self.slice_dim_low = M // num_slices
        self.slice_dim_high = (3 * M) // num_slices

        # 1. Feature Extractor (Vision Mamba + MoE)
        self.g_a = FrequencySplitEncoder(M=M)
        self.g_s = FrequencySplitDecoder(M=M)

        # 2. Latent Disentanglement
        self.latent_dwt = LatentDWT(wave='haar')
        self.latent_idwt = LatentIDWT(wave='haar')

        # 3. Hyper-Prior (Processes 4*M concatenated channels)
        self.h_a = nn.Sequential(
            ConvBottleneckBlockWithStride(4 * M, N), MambaBlock(N), nn.Conv2d(N, N, 3, 2, 1)
        )
        self.h_mean_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, 3, 2, 1, 1), MambaBlock(N), ConvBottleneckBlockWithUpsample(N, M)
        )
        self.h_scale_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, 3, 2, 1, 1), MambaBlock(N), ConvBottleneckBlockWithUpsample(N, M)
        )

        # 4. Global Dataset Dictionary
        dict_tokens = 128
        dict_dim = 256
        self.global_dictionary = nn.Parameter(torch.randn(1, dict_tokens, dict_dim))
        nn.init.trunc_normal_(self.global_dictionary, std=0.02)

        # 5. Two-Stage Dictionary Context Models
        self.dict_atten_low, self.dict_atten_high = nn.ModuleList(), nn.ModuleList()
        self.mean_low, self.mean_high = nn.ModuleList(), nn.ModuleList()
        self.scale_low, self.scale_high = nn.ModuleList(), nn.ModuleList()
        self.lrp_low, self.lrp_high = nn.ModuleList(), nn.ModuleList()

        for i in range(num_slices):
            # A. Low Frequency Branch (Condition: Hyper-Prior + Prior Low Slices)
            in_ch_low = (2 * M) + (i * self.slice_dim_low)
            self.dict_atten_low.append(DictionaryCrossAttentionContext(in_ch_low, 128, dict_dim))
            fused_low = in_ch_low + 128
            self.mean_low.append(nn.Sequential(nn.Conv2d(fused_low, 256, 3, 1, 1), nn.GELU(), nn.Conv2d(256, self.slice_dim_low, 3, 1, 1)))
            self.scale_low.append(nn.Sequential(nn.Conv2d(fused_low, 256, 3, 1, 1), nn.GELU(), nn.Conv2d(256, self.slice_dim_low, 3, 1, 1)))
            self.lrp_low.append(nn.Sequential(nn.Conv2d(fused_low + self.slice_dim_low, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, self.slice_dim_low, 3, 1, 1)))

            # B. High Frequency Branch (Condition: Hyper-Prior + All Low + Prior High Slices)
            in_ch_high = (2 * M) + M + (i * self.slice_dim_high)
            self.dict_atten_high.append(DictionaryCrossAttentionContext(in_ch_high, 128, dict_dim))
            fused_high = in_ch_high + 128
            self.mean_high.append(nn.Sequential(nn.Conv2d(fused_high, 256, 3, 1, 1), nn.GELU(), nn.Conv2d(256, self.slice_dim_high, 3, 1, 1)))
            self.scale_high.append(nn.Sequential(nn.Conv2d(fused_high, 256, 3, 1, 1), nn.GELU(), nn.Conv2d(256, self.slice_dim_high, 3, 1, 1)))
            self.lrp_high.append(nn.Sequential(nn.Conv2d(fused_high + self.slice_dim_high, 128, 3, 1, 1), nn.GELU(), nn.Conv2d(128, self.slice_dim_high, 3, 1, 1)))

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gc_low = GaussianConditional(None)
        self.gc_high = GaussianConditional(None)

    def get_moe_modules(self):
        # Used by the train.py LossFreeBalancer
        return [self.g_a.hf_proj[1], self.g_s.hf_recon[0]]

    def update(self, scale_table=None, force=False):
        if scale_table is None: scale_table = get_scale_table()
        updated = self.gc_low.update_scale_table(scale_table, force=force)
        updated |= self.gc_high.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, training_mode="noise"):
        B = x.size(0)
        y, enc_router_data = self.g_a(x)

        # Latent DWT Separation
        y_dwt = self.latent_dwt(y)
        y_low = y_dwt[:, :self.M, :, :]      
        y_high = y_dwt[:, self.M:, :, :]     
        y_shape_dwt = y_low.shape[2:]

        # Hyper-Prior
        z = self.h_a(y_dwt)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        batch_dict = self.global_dictionary.expand(B, -1, -1)

        # --- Stage A: Low Frequency Decoding ---
        y_low_slices = y_low.chunk(self.num_slices, 1)
        y_low_hat_slices, ll_low = [], []

        for i in range(self.num_slices):
            ctx = torch.cat([latent_means, latent_scales] + y_low_hat_slices, dim=1)
            fused = torch.cat([ctx, self.dict_atten_low[i](ctx, batch_dict)], dim=1)
            mu = self.mean_low[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]
            scale = self.scale_low[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]

            _, y_slice_lik = self.gc_low(y_low_slices[i], scale, mu)
            ll_low.append(y_slice_lik)
            
            y_hat_slice = y_low_slices[i] + torch.empty_like(y_low_slices[i]).uniform_(-0.5, 0.5) if (self.training and training_mode == "noise") else ste_round(y_low_slices[i] - mu) + mu
            lrp = self.lrp_low[i](torch.cat([fused, y_hat_slice], dim=1))
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_low_hat_slices.append(y_hat_slice)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        # --- Stage B: High Frequency Decoding ---
        y_high_slices = y_high.chunk(self.num_slices, 1)
        y_high_hat_slices, ll_high = [], []

        for i in range(self.num_slices):
            ctx = torch.cat([latent_means, latent_scales, y_low_hat] + y_high_hat_slices, dim=1)
            fused = torch.cat([ctx, self.dict_atten_high[i](ctx, batch_dict)], dim=1)
            mu = self.mean_high[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]
            scale = self.scale_high[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]

            _, y_slice_lik = self.gc_high(y_high_slices[i], scale, mu)
            ll_high.append(y_slice_lik)

            y_hat_slice = y_high_slices[i] + torch.empty_like(y_high_slices[i]).uniform_(-0.5, 0.5) if (self.training and training_mode == "noise") else ste_round(y_high_slices[i] - mu) + mu
            lrp = self.lrp_high[i](torch.cat([fused, y_hat_slice], dim=1))
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_high_hat_slices.append(y_hat_slice)

        y_high_hat = torch.cat(y_high_hat_slices, dim=1)

        # Inverse Transform & Image Reconstruction
        y_hat_dwt = torch.cat([y_low_hat, y_high_hat], dim=1)
        y_hat = self.latent_idwt(y_hat_dwt)
        x_hat, dec_router_data = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y_low": torch.cat(ll_low, dim=1), "y_high": torch.cat(ll_high, dim=1), "z": z_likelihoods},
            "router_logits": [enc_router_data, dec_router_data],
        }

    def compress(self, x):
        B = x.size(0)
        y, _ = self.g_a(x)
        y_dwt = self.latent_dwt(y)
        y_low, y_high = y_dwt[:, :self.M, :, :], y_dwt[:, self.M:, :, :]
        y_shape_dwt = y_low.shape[2:]

        z = self.h_a(y_dwt)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales, latent_means = self.h_scale_s(z_hat), self.h_mean_s(z_hat)
        
        batch_dict = self.global_dictionary.expand(B, -1, -1)

        # Encode Stage A (Low)
        enc_low = BufferedRansEncoder()
        sym_low, idx_low, y_low_hat_slices = [], [], []
        for i, y_slice in enumerate(y_low.chunk(self.num_slices, 1)):
            ctx = torch.cat([latent_means, latent_scales] + y_low_hat_slices, dim=1)
            fused = torch.cat([ctx, self.dict_atten_low[i](ctx, batch_dict)], dim=1)
            mu = self.mean_low[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]
            scale = self.scale_low[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]

            index = self.gc_low.build_indexes(scale)
            y_q = self.gc_low.quantize(y_slice, "symbols", mu)
            sym_low.extend(y_q.reshape(-1).tolist())
            idx_low.extend(index.reshape(-1).tolist())

            y_hat_slice = y_q + mu
            y_hat_slice += 0.5 * torch.tanh(self.lrp_low[i](torch.cat([fused, y_hat_slice], dim=1)))
            y_low_hat_slices.append(y_hat_slice)
            
        y_low_hat = torch.cat(y_low_hat_slices, dim=1)
        enc_low.encode_with_indexes(sym_low, idx_low, self.gc_low.quantized_cdf.tolist(), self.gc_low.cdf_length.reshape(-1).int().tolist(), self.gc_low.offset.reshape(-1).int().tolist())

        # Encode Stage B (High)
        enc_high = BufferedRansEncoder()
        sym_high, idx_high, y_high_hat_slices = [], [], []
        for i, y_slice in enumerate(y_high.chunk(self.num_slices, 1)):
            ctx = torch.cat([latent_means, latent_scales, y_low_hat] + y_high_hat_slices, dim=1)
            fused = torch.cat([ctx, self.dict_atten_high[i](ctx, batch_dict)], dim=1)
            mu = self.mean_high[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]
            scale = self.scale_high[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]

            index = self.gc_high.build_indexes(scale)
            y_q = self.gc_high.quantize(y_slice, "symbols", mu)
            sym_high.extend(y_q.reshape(-1).tolist())
            idx_high.extend(index.reshape(-1).tolist())

            y_hat_slice = y_q + mu
            y_hat_slice += 0.5 * torch.tanh(self.lrp_high[i](torch.cat([fused, y_hat_slice], dim=1)))
            y_high_hat_slices.append(y_hat_slice)

        enc_high.encode_with_indexes(sym_high, idx_high, self.gc_high.quantized_cdf.tolist(), self.gc_high.cdf_length.reshape(-1).int().tolist(), self.gc_high.offset.reshape(-1).int().tolist())

        return {"strings": [[enc_low.flush()], [enc_high.flush()], z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)
        B = z_hat.size(0)
        latent_scales, latent_means = self.h_scale_s(z_hat), self.h_mean_s(z_hat)
        y_shape_dwt = [z_hat.shape[2] * 2, z_hat.shape[3] * 2] # Up-sampled by 2
        batch_dict = self.global_dictionary.expand(B, -1, -1)

        # Decode Stage A (Low)
        dec_low = RansDecoder()
        dec_low.set_stream(strings[0][0])
        y_low_hat_slices = []
        for i in range(self.num_slices):
            ctx = torch.cat([latent_means, latent_scales] + y_low_hat_slices, dim=1)
            fused = torch.cat([ctx, self.dict_atten_low[i](ctx, batch_dict)], dim=1)
            mu = self.mean_low[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]
            scale = self.scale_low[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]

            index = self.gc_low.build_indexes(scale)
            rv = dec_low.decode_stream(index.reshape(-1).tolist(), self.gc_low.quantized_cdf.tolist(), self.gc_low.cdf_length.reshape(-1).int().tolist(), self.gc_low.offset.reshape(-1).int().tolist())
            y_hat_slice = self.gc_low.dequantize(torch.tensor(rv, dtype=torch.float32, device=mu.device).reshape(1, self.slice_dim_low, y_shape_dwt[0], y_shape_dwt[1]), mu)
            y_hat_slice += 0.5 * torch.tanh(self.lrp_low[i](torch.cat([fused, y_hat_slice], dim=1)))
            y_low_hat_slices.append(y_hat_slice)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        # Decode Stage B (High)
        dec_high = RansDecoder()
        dec_high.set_stream(strings[1][0])
        y_high_hat_slices = []
        for i in range(self.num_slices):
            ctx = torch.cat([latent_means, latent_scales, y_low_hat] + y_high_hat_slices, dim=1)
            fused = torch.cat([ctx, self.dict_atten_high[i](ctx, batch_dict)], dim=1)
            mu = self.mean_high[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]
            scale = self.scale_high[i](fused)[:, :, :y_shape_dwt[0], :y_shape_dwt[1]]

            index = self.gc_high.build_indexes(scale)
            rv = dec_high.decode_stream(index.reshape(-1).tolist(), self.gc_high.quantized_cdf.tolist(), self.gc_high.cdf_length.reshape(-1).int().tolist(), self.gc_high.offset.reshape(-1).int().tolist())
            y_hat_slice = self.gc_high.dequantize(torch.tensor(rv, dtype=torch.float32, device=mu.device).reshape(1, self.slice_dim_high, y_shape_dwt[0], y_shape_dwt[1]), mu)
            y_hat_slice += 0.5 * torch.tanh(self.lrp_high[i](torch.cat([fused, y_hat_slice], dim=1)))
            y_high_hat_slices.append(y_hat_slice)

        y_high_hat = torch.cat(y_high_hat_slices, dim=1)

        # Reconstruct
        y_hat = self.latent_idwt(torch.cat([y_low_hat, y_high_hat], dim=1))
        x_hat, _ = self.g_s(y_hat)
        return {"x_hat": x_hat.clamp(0, 1)}

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(self.gc_low, "gc_low", ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        update_registered_buffers(self.gc_high, "gc_high", ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict)
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls(N=192, M=320, num_slices=5)
        net.load_state_dict(state_dict)
        return net