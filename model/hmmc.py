import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# PART 1: HELPER FUNCTIONS & WAVELET (DWT)
# ==========================================

def ste_round(x):
    return torch.round(x) - x.detach() + x

def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class HaarDWT(nn.Module):
    def forward(self, x):
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) / 2.0
        HL = (x00 - x01 + x10 - x11) / 2.0
        LH = (x00 + x01 - x10 - x11) / 2.0
        HH = (x00 - x01 - x10 + x11) / 2.0
        HF = torch.cat([HL, LH, HH], dim=1)
        return LL, HF

class HaarIDWT(nn.Module):
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

# ==========================================
# PART 2: MAMBA BACKBONE & SPATIAL MoE
# ==========================================

class MambaBlock(nn.Module):
    def __init__(self, dim, ssm_ratio=2.0, d_state=16, dt_rank="auto", d_conv=3, drop_path=0.0):
        super().__init__()
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
        self.dwt = HaarDWT()
        self.ll_proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            ConvNeXtBlock(64), ConvNeXtBlock(64)
        )
        self.hf_proj = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=5, stride=2, padding=2),
            SpatialSparseMoE(dim=64, num_experts=4)
        )
        self.merge = nn.Sequential(
            nn.Conv2d(128, 192, 1),
            ConvBottleneckBlockWithStride(192, 256),
            MambaBlockSequence(256, 256, num_blocks=3),
            nn.Conv2d(256, M, kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x):
        LL, HF = self.dwt(x)
        ll_feat = self.ll_proj(LL)
        hf_feat, router_data = self.hf_proj[1](self.hf_proj[0](HF))
        y = self.merge(torch.cat([ll_feat, hf_feat], dim=1))
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
            ConvNeXtBlock(64), ConvNeXtBlock(64),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        self.hf_recon = nn.Sequential(
            SpatialSparseMoE(dim=64, num_experts=4),
            nn.ConvTranspose2d(64, 9, kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        self.idwt = HaarIDWT()

    def forward(self, y):
        ll_feat, hf_feat = torch.split(self.split(y), 64, dim=1)
        LL_hat = self.ll_recon(ll_feat)
        hf_moe_out, router_data = self.hf_recon[0](hf_feat)
        HF_hat = self.hf_recon[1](hf_moe_out)
        return self.idwt(LL_hat, HF_hat), router_data

# ==========================================
# PART 3: DCAE DICTIONARY CROSS-ATTENTION
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
    """
    SOTA Context Predictor: Queries a Global Dataset Dictionary 
    using local spatial context from hyper-prior & previous channel slices.
    """
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
        
        # Query (Image Context)
        q = self.q_proj(context)
        q = rearrange(q, 'b c h w -> b (h w) c')
        q = self.norm_q(q)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Key/Value (Global Dictionary)
        dict_norm = self.norm_k(global_dict)
        k = rearrange(self.k_proj(dict_norm), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(dict_norm, 'b t (h d) -> b h t d', h=self.num_heads)
        
        # Cross Attention
        attn = torch.einsum('bhnd, bhtd -> bhnt', q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        dict_features = torch.einsum('bhnt, bhtd -> bhnd', attn, v)
        dict_features = rearrange(dict_features, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        
        return self.glu(self.out_proj(dict_features))

# ==========================================
# PART 4: INTEGRATED SOTA ARCHITECTURE (HMMC + DCAE)
# ==========================================

class HMMC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.slice_dim = M // num_slices

        # 1. HMMC Frequency Disentangled Backbone
        self.g_a = FrequencySplitEncoder(M=M)
        self.g_s = FrequencySplitDecoder(M=M)

        # 2. Hyper-Prior Model (Standard)
        self.h_a = nn.Sequential(
            ConvBottleneckBlockWithStride(M, N),
            MambaBlock(N),
            nn.Conv2d(N, 192, kernel_size=3, stride=2, padding=1),
        )
        self.h_z_s1 = nn.Sequential(
            nn.ConvTranspose2d(192, N, kernel_size=3, stride=2, output_padding=1, padding=1),
            MambaBlock(N), ConvBottleneckBlockWithUpsample(N, M),
        )
        self.h_z_s2 = nn.Sequential(
            nn.ConvTranspose2d(192, N, kernel_size=3, stride=2, output_padding=1, padding=1),
            MambaBlock(N), ConvBottleneckBlockWithUpsample(N, M),
        )

        # 3. DCAE Global Dictionary
        dict_tokens = 128
        dict_dim = 256
        self.global_dictionary = nn.Parameter(torch.randn(1, dict_tokens, dict_dim))
        nn.init.trunc_normal_(self.global_dictionary, std=0.02)

        # 4. Channel-Conditional Dictionary Transforms
        self.dict_attentions = nn.ModuleList()
        self.mean_transforms = nn.ModuleList()
        self.scale_transforms = nn.ModuleList()
        self.lrp_transforms = nn.ModuleList()

        for i in range(num_slices):
            # hyper_means + hyper_scales + previously decoded slices
            in_channels = (2 * M) + (i * self.slice_dim) 
            
            self.dict_attentions.append(DictionaryCrossAttentionContext(in_dim=in_channels, out_dim=128, dict_dim=dict_dim))
            
            fused_dim = in_channels + 128 # original context + retrieved dict features
            self.mean_transforms.append(nn.Sequential(
                nn.Conv2d(fused_dim, 256, 3, padding=1), nn.GELU(), nn.Conv2d(256, self.slice_dim, 3, padding=1)
            ))
            self.scale_transforms.append(nn.Sequential(
                nn.Conv2d(fused_dim, 256, 3, padding=1), nn.GELU(), nn.Conv2d(256, self.slice_dim, 3, padding=1)
            ))
            self.lrp_transforms.append(nn.Sequential(
                nn.Conv2d(fused_dim + self.slice_dim, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, self.slice_dim, 3, padding=1)
            ))

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def get_moe_modules(self):
        return [self.g_a.hf_proj[1], self.g_s.hf_recon[0]]

    def update(self, scale_table=None, force=False):
        if scale_table is None: scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x, training_mode="noise"):
        B = x.size(0)
        y, enc_router_data = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        
        batch_dict = self.global_dictionary.expand(B, -1, -1)
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices, y_likelihoods_list = [], []

        for i in range(self.num_slices):
            y_slice = y_slices[i]
            
            context = torch.cat([latent_means, latent_scales] + y_hat_slices, dim=1)
            dict_features = self.dict_attentions[i](context, batch_dict)
            fused_support = torch.cat([context, dict_features], dim=1)

            mu = self.mean_transforms[i](fused_support)[:, :, :y_shape[0], :y_shape[1]]
            scale = self.scale_transforms[i](fused_support)[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihoods_list.append(y_slice_likelihood)

            if self.training and training_mode == "noise":
                y_hat_slice = y_slice + torch.empty_like(y_slice).uniform_(-0.5, 0.5)
            else:
                y_hat_slice = ste_round(y_slice - mu) + mu

            lrp = self.lrp_transforms[i](torch.cat([fused_support, y_hat_slice], dim=1))
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods_list, dim=1)
        
        x_hat, dec_router_data = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "router_logits": [enc_router_data, dec_router_data],
        }

    def compress(self, x):
        B = x.size(0)
        y, _ = self.g_a(x)
        y_shape = y.shape[2:]
        
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        
        batch_dict = self.global_dictionary.expand(B, -1, -1)
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list, indexes_list = [], []

        for i in range(self.num_slices):
            y_slice = y_slices[i]
            context = torch.cat([latent_means, latent_scales] + y_hat_slices, dim=1)
            dict_features = self.dict_attentions[i](context, batch_dict)
            fused_support = torch.cat([context, dict_features], dim=1)

            mu = self.mean_transforms[i](fused_support)[:, :, :y_shape[0], :y_shape[1]]
            scale = self.scale_transforms[i](fused_support)[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp = self.lrp_transforms[i](torch.cat([fused_support, y_hat_slice], dim=1))
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        
        return {"strings": [[y_string], z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B = z_hat.size(0)
        latent_scales = self.h_z_s1(z_hat)
        latent_means = self.h_z_s2(z_hat)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        batch_dict = self.global_dictionary.expand(B, -1, -1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])

        for i in range(self.num_slices):
            context = torch.cat([latent_means, latent_scales] + y_hat_slices, dim=1)
            dict_features = self.dict_attentions[i](context, batch_dict)
            fused_support = torch.cat([context, dict_features], dim=1)

            mu = self.mean_transforms[i](fused_support)[:, :, :y_shape[0], :y_shape[1]]
            scale = self.scale_transforms[i](fused_support)[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.tensor(rv, dtype=torch.float32, device=mu.device).reshape(1, self.slice_dim, y_shape[0], y_shape[1])
            
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp = self.lrp_transforms[i](torch.cat([fused_support, y_hat_slice], dim=1))
            y_hat_slice = y_hat_slice + (0.5 * torch.tanh(lrp))
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat, _ = self.g_s(y_hat)
        
        return {"x_hat": x_hat.clamp(0, 1)}

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional, "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"], state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_state_dict(cls, state_dict):
        net = cls(N=192, M=320, num_slices=5)
        net.load_state_dict(state_dict)
        return net