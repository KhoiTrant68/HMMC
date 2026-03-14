import math

import torch
import torch.nn as nn

try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RateDistortionLoss(nn.Module):
    def __init__(
        self, lmbda=1e-2, loss_type="mse", num_experts=4, use_loss_free_balancing=True
    ):
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type.lower().replace("-", "_")
        self.num_experts = num_experts
        self.use_loss_free_balancing = use_loss_free_balancing
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        total_bits = 0
        for name, likelihood in output["likelihoods"].items():
            if isinstance(likelihood, (list, tuple)):
                for l in likelihood:
                    total_bits += torch.log(l.clamp(min=1e-9)).sum()
            else:
                total_bits += torch.log(likelihood.clamp(min=1e-9)).sum()

        out["bpp_loss"] = -total_bits / (math.log(2) * num_pixels)
        x_hat = output["x_hat"].clamp(0, 1)
        mse_val = self.mse(x_hat, target)
        out["mse_loss"] = mse_val

        if self.loss_type == "mse":
            dist_loss = 255**2 * mse_val
        elif self.loss_type == "ms_ssim":
            out["ms_ssim_loss"] = 1 - ms_ssim(x_hat, target, data_range=1.0)
            dist_loss = out["ms_ssim_loss"]

        out["dist_loss"] = dist_loss
        out["psnr"] = (
            -10 * torch.log10(mse_val)
            if mse_val > 1e-10
            else torch.tensor(100.0, device=target.device)
        )

        if "router_logits" in output and output["router_logits"] is not None:
            with torch.no_grad():
                out["moe_imbalance"] = self._calculate_imbalance(
                    output["router_logits"]
                )

        out["loss"] = out["bpp_loss"] + self.lmbda * dist_loss
        return out

    def _calculate_imbalance(self, router_data):
        total_imbalance, count = 0.0, 0
        for item in router_data:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                indices = item[1]
                expert_counts = torch.bincount(
                    indices.flatten(), minlength=self.num_experts
                ).float()
                avg_load = expert_counts.mean()
                if avg_load > 0:
                    total_imbalance += (
                        expert_counts - avg_load
                    ).abs().mean() / avg_load
                    count += 1
        return total_imbalance / count if count > 0 else 0.0
