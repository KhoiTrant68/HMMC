import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WindowAttention(nn.Module):
    """
    Window-based Local Attention (WLA) 
    Captures local high-frequency spatial redundancy.
    """
    def __init__(self, dim, window_size=8, num_heads=4, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        # Input: (B, C, H, W)
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1) # NHWC

        # Pad to multiple of window_size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Partition
        x_windows = self.window_partition(x) 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C) 

        # Attention
        Bw, N, _ = x_windows.shape
        qkv = self.qkv(x_windows).reshape(Bw, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reverse
        x = x.view(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x, Hp, Wp)

        # Crop padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.permute(0, 3, 1, 2)
        return x + shortcut

class FlashGMMConditional(nn.Module):
    """
    Gaussian Mixture Model (K=3) with dynamic binary-search decoding.
    """
    def __init__(self, num_mixtures=3):
        super().__init__()
        self.num_mixtures = num_mixtures

    def forward(self, inputs, weights, means, scales):
        """
        Training: Calculates Probability Mass (Likelihood).
        inputs: (B, C, H, W)
        weights, means, scales: (B, K, C, H, W)
        """
        # Expand inputs: (B, 1, C, H, W)
        inputs = inputs.unsqueeze(1)
        
        # Calculate CDF for inputs+0.5 and inputs-0.5
        # Standardized: (x - mu) / (sigma * sqrt(2))
        inv_std = 1.0 / (scales * math.sqrt(2))
        
        values_upper = (inputs + 0.5 - means) * inv_std
        values_lower = (inputs - 0.5 - means) * inv_std
        
        upper_cdf = 0.5 * (1 + torch.erf(values_upper))
        lower_cdf = 0.5 * (1 + torch.erf(values_lower))
        
        # Prob = CDF(upper) - CDF(lower)
        probs = upper_cdf - lower_cdf
        probs = torch.clamp(probs, min=1e-9) 
        
        # Weighted sum: sum(w_k * p_k)
        weighted_probs = torch.sum(probs * weights, dim=1) 
        
        return weighted_probs

    def decompress(self, target_cdf, weights, means, scales):
        """
        Inference: Finds integer x such that CDF(x) ~= target_cdf using Binary Search.
        """
        # Heuristic bounds: mean +/- 20*sigma
        avg_mean = torch.sum(weights * means, dim=1)
        max_scale = torch.max(scales, dim=1)[0]
        
        lower = torch.floor(avg_mean - 20 * max_scale).int()
        upper = torch.ceil(avg_mean + 20 * max_scale).int()
        
        # 10 iterations usually sufficient for integer convergence
        for _ in range(10):
            mid = (lower + upper) // 2
            
            # CDF(mid) calculation using Fast TanH approx
            val_norm = (mid.unsqueeze(1) - means) / scales
            cdf_k = 0.5 * (1 + torch.tanh(math.sqrt(2 / math.pi) * (val_norm + 0.044715 * torch.pow(val_norm, 3))))
            
            cdf_val = torch.sum(weights * cdf_k, dim=1)
            
            mask = cdf_val < target_cdf
            lower = torch.where(mask, mid, lower)
            upper = torch.where(mask, upper, mid)
            
        return upper.float()