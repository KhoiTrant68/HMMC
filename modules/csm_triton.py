import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:

    @triton.jit
    def triton_cross_scan(
        x,
        y,
        BC: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
        DC: tl.constexpr,
        DH: tl.constexpr,
        DW: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
    ):
        i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_h, i_w = (i_hw // NW), (i_hw % NW)
        _mask_hw = ((i_h * BH + tl.arange(0, BH)) < DH)[:, None] & (
            (i_w * BW + tl.arange(0, BW)) < DW
        )[None, :]
        _for_C = min(DC - i_c * BC, BC)

        _tmp0 = i_c * BC * DH * DW
        _tmp1 = DC * DH * DW
        _tmp2 = (
            _tmp0
            + i_h * BH * DW
            + tl.arange(0, BH)[:, None] * DW
            + i_w * BW
            + tl.arange(0, BW)[None, :]
        )
        p_x = x + i_b * _tmp1 + _tmp2
        p_y1 = y + i_b * 4 * _tmp1 + _tmp2
        p_y2 = (
            y
            + i_b * 4 * _tmp1
            + _tmp1
            + _tmp0
            + i_w * BW * DH
            + tl.arange(0, BW)[None, :] * DH
            + i_h * BH
            + tl.arange(0, BH)[:, None]
        )
        p_y3 = (
            y
            + i_b * 4 * _tmp1
            + 2 * _tmp1
            + _tmp0
            + (NH - i_h - 1) * BH * DW
            + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
            + (NW - i_w - 1) * BW
            + (BW - 1 - tl.arange(0, BW)[None, :])
            + (DH - NH * BH) * DW
            + (DW - NW * BW)
        )
        p_y4 = (
            y
            + i_b * 4 * _tmp1
            + 3 * _tmp1
            + _tmp0
            + (NW - i_w - 1) * BW * DH
            + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
            + (NH - i_h - 1) * BH
            + (BH - 1 - tl.arange(0, BH)[:, None])
            + (DH - NH * BH)
            + (DW - NW * BW) * DH
        )

        for idxc in range(_for_C):
            _idx = idxc * DH * DW
            _x = tl.load(p_x + _idx, mask=_mask_hw)
            tl.store(p_y1 + _idx, _x, mask=_mask_hw)
            tl.store(p_y2 + _idx, _x, mask=_mask_hw)
            tl.store(p_y3 + _idx, _x, mask=_mask_hw)
            tl.store(p_y4 + _idx, _x, mask=_mask_hw)

    @triton.jit
    def triton_cross_merge(
        x,
        y,
        BC: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
        DC: tl.constexpr,
        DH: tl.constexpr,
        DW: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
    ):
        i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_h, i_w = (i_hw // NW), (i_hw % NW)
        _mask_hw = ((i_h * BH + tl.arange(0, BH)) < DH)[:, None] & (
            (i_w * BW + tl.arange(0, BW)) < DW
        )[None, :]
        _for_C = min(DC - i_c * BC, BC)

        _tmp0, _tmp1 = i_c * BC * DH * DW, DC * DH * DW
        _tmp2 = (
            _tmp0
            + i_h * BH * DW
            + tl.arange(0, BH)[:, None] * DW
            + i_w * BW
            + tl.arange(0, BW)[None, :]
        )
        p_x, p_y1 = x + i_b * _tmp1 + _tmp2, y + i_b * 4 * _tmp1 + _tmp2
        p_y2 = (
            y
            + i_b * 4 * _tmp1
            + _tmp1
            + _tmp0
            + i_w * BW * DH
            + tl.arange(0, BW)[None, :] * DH
            + i_h * BH
            + tl.arange(0, BH)[:, None]
        )
        p_y3 = (
            y
            + i_b * 4 * _tmp1
            + 2 * _tmp1
            + _tmp0
            + (NH - i_h - 1) * BH * DW
            + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
            + (NW - i_w - 1) * BW
            + (BW - 1 - tl.arange(0, BW)[None, :])
            + (DH - NH * BH) * DW
            + (DW - NW * BW)
        )
        p_y4 = (
            y
            + i_b * 4 * _tmp1
            + 3 * _tmp1
            + _tmp0
            + (NW - i_w - 1) * BW * DH
            + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
            + (NH - i_h - 1) * BH
            + (BH - 1 - tl.arange(0, BH)[:, None])
            + (DH - NH * BH)
            + (DW - NW * BW) * DH
        )

        for idxc in range(_for_C):
            _idx = idxc * DH * DW
            tl.store(
                p_x + _idx,
                tl.load(p_y1 + _idx, mask=_mask_hw)
                + tl.load(p_y2 + _idx, mask=_mask_hw)
                + tl.load(p_y3 + _idx, mask=_mask_hw)
                + tl.load(p_y4 + _idx, mask=_mask_hw),
                mask=_mask_hw,
            )

    @triton.jit
    def triton_cross_scan_1b1(
        x,
        y,
        BC: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
        DC: tl.constexpr,
        DH: tl.constexpr,
        DW: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
    ):
        i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_h, i_w = (i_hw // NW), (i_hw % NW)
        _mask_hw = ((i_h * BH + tl.arange(0, BH)) < DH)[:, None] & (
            (i_w * BW + tl.arange(0, BW)) < DW
        )[None, :]
        _for_C = min(DC - i_c * BC, BC)

        _tmp0, _tmp1 = i_c * BC * DH * DW, DC * DH * DW
        _tmp2 = (
            _tmp0
            + i_h * BH * DW
            + tl.arange(0, BH)[:, None] * DW
            + i_w * BW
            + tl.arange(0, BW)[None, :]
        )
        p_y1 = y + i_b * 4 * _tmp1 + _tmp2
        p_y2 = (
            y
            + i_b * 4 * _tmp1
            + _tmp1
            + _tmp0
            + i_w * BW * DH
            + tl.arange(0, BW)[None, :] * DH
            + i_h * BH
            + tl.arange(0, BH)[:, None]
        )
        p_y3 = (
            y
            + i_b * 4 * _tmp1
            + 2 * _tmp1
            + _tmp0
            + (NH - i_h - 1) * BH * DW
            + (BH - 1 - tl.arange(0, BH)[:, None]) * DW
            + (NW - i_w - 1) * BW
            + (BW - 1 - tl.arange(0, BW)[None, :])
            + (DH - NH * BH) * DW
            + (DW - NW * BW)
        )
        p_y4 = (
            y
            + i_b * 4 * _tmp1
            + 3 * _tmp1
            + _tmp0
            + (NW - i_w - 1) * BW * DH
            + (BW - 1 - tl.arange(0, BW)[None, :]) * DH
            + (NH - i_h - 1) * BH
            + (BH - 1 - tl.arange(0, BH)[:, None])
            + (DH - NH * BH)
            + (DW - NW * BW) * DH
        )
        p_x1 = x + i_b * 4 * _tmp1 + _tmp2

        for idxc in range(_for_C):
            _idx = idxc * DH * DW
            tl.store(p_y1 + _idx, tl.load(p_x1 + _idx), mask=_mask_hw)
            tl.store(p_y2 + _idx, tl.load(p_x1 + _tmp1 + _idx), mask=_mask_hw)
            tl.store(p_y3 + _idx, tl.load(p_x1 + 2 * _tmp1 + _idx), mask=_mask_hw)
            tl.store(p_y4 + _idx, tl.load(p_x1 + 3 * _tmp1 + _idx), mask=_mask_hw)

    class CrossScanTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = int(x.size(0)), int(x.size(1)), int(x.size(2)), int(x.size(3))
            BC, BH, BW = (
                min(triton.next_power_of_2(C), 2),
                min(triton.next_power_of_2(H), 32),
                min(triton.next_power_of_2(W), 32),
            )
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.shape, ctx.triton_shape = (B, C, H, W), (BC, BH, BW, NC, NH, NW)
            y = x.contiguous().new_empty((B, 4, C, H, W))
            triton_cross_scan[(NH * NW, NC, B)](
                x.contiguous(), y, BC, BH, BW, C, H, W, NH, NW
            )
            return y.view(B, 4, C, -1)

        @staticmethod
        def backward(ctx, y: torch.Tensor):
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            x = y.contiguous().new_empty((B, C, H, W))
            triton_cross_merge[(NH * NW, NC, B)](
                x, y.contiguous().view(B, 4, C, H, W), BC, BH, BW, C, H, W, NH, NW
            )
            return x

    class CrossMergeTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y: torch.Tensor):
            B, K, C, H, W = y.shape
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = (
                min(triton.next_power_of_2(C), 2),
                min(triton.next_power_of_2(H), 32),
                min(triton.next_power_of_2(W), 32),
            )
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.shape, ctx.triton_shape = (B, C, H, W), (BC, BH, BW, NC, NH, NW)
            x = y.contiguous().new_empty((B, C, H, W))
            triton_cross_merge[(NH * NW, NC, B)](
                x, y.contiguous().view(B, 4, C, H, W), BC, BH, BW, C, H, W, NH, NW
            )
            return x.view(B, C, -1)

        @staticmethod
        def backward(ctx, x: torch.Tensor):
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            y = x.contiguous().new_empty((B, 4, C, H, W))
            triton_cross_scan[(NH * NW, NC, B)](
                x.contiguous(), y, BC, BH, BW, C, H, W, NH, NW
            )
            return y

    class CrossScanTriton1b1(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, K, C, H, W = x.shape
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = (
                min(triton.next_power_of_2(C), 2),
                min(triton.next_power_of_2(H), 32),
                min(triton.next_power_of_2(W), 32),
            )
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            y = x.contiguous().new_empty((B, 4, C, H, W))
            triton_cross_scan_1b1[(NH * NW, NC, B)](
                x.contiguous(), y, BC, BH, BW, C, H, W, NH, NW
            )
            return y.view(B, 4, C, -1)

        @staticmethod
        def backward(ctx, y: torch.Tensor):
            raise NotImplementedError

else:
    # PYTORCH FALLBACK FOR TRITON
    class CrossScanTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs

        @staticmethod
        def backward(ctx, ys):
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
                dim0=2, dim1=3
            ).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)

    class CrossMergeTriton(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(
                dim0=2, dim1=3
            ).contiguous().view(B, D, -1)
            return y

        @staticmethod
        def backward(ctx, x):
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs.view(B, 4, C, H, W)

    class CrossScanTriton1b1(CrossScanTriton):
        pass
