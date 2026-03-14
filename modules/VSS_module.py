import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from modules.csm_triton import CrossMergeTriton, CrossScanTriton, CrossScanTriton1b1

try:
    import selective_scan_cuda
    import selective_scan_cuda_core
    import selective_scan_cuda_oflex
except Exception:
    pass


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
            dim0=2, dim1=3
        ).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs.view(B, 4, C, H, W)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(
        ctx,
        u,
        delta,
        A,
        B,
        C,
        D=None,
        delta_bias=None,
        delta_softplus=False,
        nrows=1,
        backnrows=1,
        oflex=True,
    ):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(
            u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex
        )
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        dout = dout.contiguous() if dout.stride(-1) != 1 else dout
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def cross_selective_scan(
    x=None,
    x_proj_weight=None,
    x_proj_bias=None,
    dt_projs_weight=None,
    dt_projs_bias=None,
    A_logs=None,
    Ds=None,
    delta_softplus=True,
    out_norm=None,
    out_norm_shape="v0",
    to_dtype=True,
    force_fp32=False,
    nrows=-1,
    backnrows=-1,
    ssoflex=True,
    SelectiveScan=None,
    CrossScan=CrossScan,
    CrossMerge=CrossMerge,
    no_einsum=False,
    dt_low_rank=True,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    xs = CrossScan.apply(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))
    Bs, Cs = Bs.contiguous().view(B, K, N, L), Cs.contiguous().view(B, K, N, L)
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs, dts, Bs, Cs = (
            xs.to(torch.float),
            dts.to(torch.float),
            Bs.to(torch.float),
            Cs.to(torch.float),
        )

    ys = SelectiveScan.apply(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, 1, 1, ssoflex
    ).view(B, K, -1, H, W)
    y = CrossMerge.apply(ys)

    if out_norm_shape == "v1":
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1)
    else:
        y = out_norm(y.transpose(dim0=1, dim1=2).contiguous()).view(B, H, W, -1)

    return y.to(x.dtype) if to_dtype else y


class SS2D(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v2",
        **kwargs
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.out_norm = nn.LayerNorm(d_inner)
        self.forward_core = partial(
            self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex
        )
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        self.x_proj_weight = nn.Parameter(
            torch.stack(
                [
                    nn.Linear(d_inner, dt_rank + d_state * 2, bias=False).weight
                    for _ in range(4)
                ],
                dim=0,
            )
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        dt_projs = []
        for _ in range(4):
            dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
            nn.init.uniform_(
                dt_proj.weight, -(dt_rank**-0.5 * dt_scale), dt_rank**-0.5 * dt_scale
            )
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            with torch.no_grad():
                dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            dt_projs.append(dt_proj)

        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_projs], dim=0)
        )
        self.A_logs = nn.Parameter(
            repeat(
                torch.log(
                    repeat(
                        torch.arange(1, d_state + 1, dtype=torch.float32),
                        "n -> d n",
                        d=d_inner,
                    )
                ),
                "d n -> r d n",
                r=4,
            ).flatten(0, 1)
        )
        self.Ds = nn.Parameter(
            repeat(torch.ones(d_inner), "n1 -> r n1", r=4).flatten(0, 1)
        )

    def forward_corev2(self, x: torch.Tensor, **kwargs):
        return cross_selective_scan(
            x,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            out_norm=self.out_norm,
            out_norm_shape="v0",
            **kwargs
        )

    def forward(self, x: torch.Tensor, **kwargs):
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        return self.dropout(self.out_proj(self.forward_core(x) * self.act(z)))
