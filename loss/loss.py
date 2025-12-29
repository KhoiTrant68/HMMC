import math
import torch
import torch.nn as nn

# Try importing MS-SSIM
try:
    from pytorch_msssim import ms_ssim
except ImportError:
    ms_ssim = None


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RateDistortionLoss(nn.Module):
    """
    Custom Rate-Distortion loss module.
    Optimized for Loss-Free Balancing based on DeepSeek-AI's strategy.
    """

    def __init__(
        self,
        lmbda=1e-2,
        loss_type="mse",
        num_experts=4,
        use_loss_free_balancing=True,
    ):
        super().__init__()
        self.lmbda = lmbda
        self.loss_type = loss_type.lower().replace("-", "_")
        self.num_experts = num_experts
        self.use_loss_free_balancing = use_loss_free_balancing

        self.mse = nn.MSELoss()

    def forward(self, output, target):
        """
        output: Dictionary from the model
        target: Original ground truth image [0, 1]
        """
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}

        # --- 1. RATE LOSS (BPP) ---
        # Calculate bits based on likelihoods for y and z latents
        total_bits = 0
        for name, likelihood in output["likelihoods"].items():
            if isinstance(likelihood, (list, tuple)):
                for l in likelihood:
                    total_bits += torch.log(l.clamp(min=1e-9)).sum()
            else:
                total_bits += torch.log(likelihood.clamp(min=1e-9)).sum()

        # BPP = -log2(likelihood) / total_pixels
        out["bpp_loss"] = -total_bits / (math.log(2) * num_pixels)

        # --- 2. DISTORTION LOSS ---
        x_hat = output["x_hat"].clamp(0, 1)

        # Initialize dist_loss
        dist_loss = torch.tensor(0.0, device=target.device)

        # Compute MSE (needed for PSNR regardless of loss_type)
        mse_val = self.mse(x_hat, target)
        out["mse_loss"] = mse_val

        if self.loss_type == "mse":
            # Scale MSE by 255^2 to align magnitude with BPP if desired, 
            # or keep raw. Standard CompressAI uses 255**2 * MSE for lambda tuning.
            dist_loss = 255**2 * mse_val

        elif self.loss_type == "ms_ssim":
            if ms_ssim is None:
                raise ImportError(
                    "pytorch_msssim not installed. Install it with: pip install pytorch-msssim"
                )
            out["ms_ssim_loss"] = 1 - ms_ssim(x_hat, target, data_range=1.0)
            dist_loss = out["ms_ssim_loss"]
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        out["dist_loss"] = dist_loss

        # --- 3. PSNR (Critical Fix) ---
        # PSNR = -10 * log10(MSE)
        # Handle case where MSE is 0 to avoid NaN
        if mse_val > 1e-10:
            out["psnr"] = -10 * torch.log10(mse_val)
        else:
            out["psnr"] = torch.tensor(100.0, device=target.device)

        # --- 4. MOE MONITORING (Loss-Free Metric) ---
        if "router_logits" in output and output["router_logits"] is not None:
            with torch.no_grad():
                out["moe_imbalance"] = self._calculate_imbalance(
                    output["router_logits"]
                )

        # --- 5. TOTAL LOSS ---
        out["loss"] = out["bpp_loss"] + self.lmbda * dist_loss

        return out

    def _calculate_imbalance(self, router_data):
        """Calculates the standard Load Imbalance metric (0.0 is perfect balance)."""
        total_imbalance = 0.0
        count = 0

        for item in router_data:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                indices = item[1]  # [B, H, W, K]

                # Count expert selections
                expert_counts = torch.bincount(
                    indices.flatten(), minlength=self.num_experts
                ).float()

                avg_load = expert_counts.mean()
                if avg_load > 0:
                    # Normalized mean absolute deviation
                    imbalance = (expert_counts - avg_load).abs().mean() / avg_load
                    total_imbalance += imbalance
                    count += 1

        return total_imbalance / count if count > 0 else 0.0