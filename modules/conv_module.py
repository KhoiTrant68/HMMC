import torch
import torch.nn as nn
from timm.layers import DropPath
from torch.nn import LayerNorm


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma, self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim)), nn.Parameter(
            torch.zeros(1, 1, 1, dim)
        )

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        return (
            self.gamma * (x * (Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)))
            + self.beta
            + x
        )


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.pwconv2(
            self.grn(
                self.act(self.pwconv1(self.norm(self.dwconv(x).permute(0, 2, 3, 1))))
            )
        )
        if self.gamma is not None:
            x = self.gamma * x
        return input + self.drop_path(x.permute(0, 3, 1, 2))


class ConvBottleneckBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 3):
        super().__init__()
        self.conv_down = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
        self.res_blocks = nn.Sequential(
            *[ConvNeXtBlock(out_ch) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.res_blocks(self.conv_down(x))


class ConvBottleneckBlockWithUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 3):
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[ConvNeXtBlock(in_ch) for _ in range(num_layers)]
        )
        self.conv_up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, x):
        return self.conv_up(self.res_blocks(x))
