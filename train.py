import argparse
import logging
import math
import os
import shutil
import sys
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR

# Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# CompressAI
from compressai.datasets import ImageFolder

# TIMM (for EMA)
from timm.utils import ModelEmaV2

# Local Imports (Ensure these files exist based on previous steps)
from model.hdmc import HDMC
from loss.loss import RateDistortionLoss, AverageMeter

# =========================================================
#  LOSS-FREE BALANCER (Mamba/MoE Optimized)
# =========================================================
class LossFreeBalancer:
    """
    Optimized for EfficientMoELayer in HDMC.
    """
    def __init__(self, update_rate=0.001):
        self.update_rate = update_rate

    @torch.no_grad()
    def update_biases(self, model, router_data_tuple, accelerator=None):
        """
        Updates expert biases to encourage equal load.
        """
        if not router_data_tuple:
            return

        # router_data_tuple structure: list of (logits, indices)
        # Order in HDMC_Mamba forward: [Slice 0...N, Anchor, Non-Anchor]
        
        # We need to map the flat list of router outputs back to the specific modules
        # This requires knowing the order in which HDMC returns them.
        # Based on hdmc.py: 
        # 1. Standard slices (model.moe_layers)
        # 2. Anchor (model.moe_anchor)
        # 3. Non-Anchor (model.moe_non_anchor)

        moe_modules = list(model.moe_layers) + [model.moe_anchor, model.moe_non_anchor]

        if len(router_data_tuple) != len(moe_modules):
            # Mismatch implies some MoEs didn't run or logic changed; skip to be safe
            return

        for i, (logits, topk_indices) in enumerate(router_data_tuple):
            target_module = moe_modules[i]
            
            # 1. Calculate Load
            # Indices shape: [B, H, W, k] -> flatten
            flat_indices = topk_indices.flatten()
            num_experts = target_module.num_experts
            
            local_counts = torch.bincount(
                flat_indices, minlength=num_experts
            ).float()

            # 2. Sync across GPUs
            if accelerator and accelerator.num_processes > 1:
                expert_counts = accelerator.reduce(local_counts, reduction="sum")
            else:
                expert_counts = local_counts

            # 3. Compute Error (Target = Average Load)
            avg_load = expert_counts.mean()
            # If count < avg, error > 0 -> bias increases
            # If count > avg, error < 0 -> bias decreases
            error = avg_load - expert_counts 

            # 4. Update
            if hasattr(target_module, "expert_biases"):
                # Normalize update by batch size magnitude to keep it stable
                # magnitude = avg_load.clamp(min=1.0)
                update_step = self.update_rate * torch.sign(error)
                
                # In-place add
                target_module.expert_biases.data.add_(update_step)

# =========================================================
#  UTILS
# =========================================================

def setup_logger(log_dir):
    logger = logging.getLogger("HDMC_Train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # Avoid duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def configure_optimizers(net, args):
    """
    Separates Main parameters (AdamW) from Entropy Parameters (Adam).
    Entropy parameters (quantiles) often require different handling.
    """
    params_dict = dict(net.named_parameters())
    
    # CompressAI models suffix quantile parameters with ".quantiles"
    aux_params = {n: p for n, p in params_dict.items() if n.endswith(".quantiles")}
    main_params = {n: p for n, p in params_dict.items() if not n.endswith(".quantiles")}

    # Main Optimizer: AdamW is generally better for Transformer/Mamba architectures
    optimizer = optim.AdamW(
        [p for p in main_params.values() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Aux Optimizer: Standard Adam for CDF updates
    aux_optimizer = optim.Adam(
        [p for p in aux_params.values() if p.requires_grad],
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer

def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, filename),
            os.path.join(save_path, "checkpoint_best.pth.tar"),
        )

# =========================================================
#  EPOCH LOOPS
# =========================================================

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer,
    epoch, accelerator, logger, writer, global_step, balancer, ema_model, args
):
    model.train()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    aux_meter = AverageMeter()

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        
        # 1. Main Forward Pass
        # training_mode="noise" is standard for training entropy models
        out_net = model(batch, training_mode="noise")
        
        # 2. Compute Loss
        out_criterion = criterion(out_net, batch)
        loss = out_criterion["loss"]
        
        # 3. Backward Main
        accelerator.backward(loss)
        
        # Clip gradients (Critical for Mamba/RNN stability)
        if args.clip_max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            
        optimizer.step()
        
        # 4. Aux Loss (Entropy Bottleneck)
        # We need the unwrapped model to access .aux_loss() specific to CompressAI
        unwrapped_model = accelerator.unwrap_model(model)
        
        aux_loss = unwrapped_model.aux_loss()
        accelerator.backward(aux_loss)
        aux_optimizer.step()
        
        # 5. Load Balancing (Loss-Free)
        # Performed after optimization step
        if "router_logits" in out_net and out_net["router_logits"]:
            balancer.update_biases(unwrapped_model, out_net["router_logits"], accelerator)
            
        # 6. EMA Update
        if ema_model is not None:
            ema_model.update(unwrapped_model)

        # 7. Logging
        loss_meter.update(loss.item())
        bpp_meter.update(out_criterion["bpp_loss"].item())
        mse_meter.update(out_criterion["mse_loss"].item())
        aux_meter.update(aux_loss.item())
        
        if i % args.print_freq == 0 and accelerator.is_main_process:
            logger.info(
                f"Train Epoch: [{epoch}][{i}/{len(train_dataloader)}] "
                f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"Bpp: {bpp_meter.val:.4f} | MSE: {mse_meter.val:.6f}"
            )
            writer.add_scalar("Train/Loss", loss_meter.val, global_step)
            writer.add_scalar("Train/Bpp", bpp_meter.val, global_step)
            writer.add_scalar("Train/MSE", mse_meter.val, global_step)
            writer.add_scalar("Train/Aux", aux_meter.val, global_step)
            
            # Debug MoE Biases occasionally
            if i % 500 == 0 and hasattr(unwrapped_model.moe_layers[0], "expert_biases"):
                b = unwrapped_model.moe_layers[0].expert_biases
                writer.add_histogram("Debug/ExpertBiases_Layer0", b, global_step)

        global_step += 1
        
    return global_step

def test_epoch(epoch, test_dataloader, model, criterion, accelerator, logger, writer):
    model.eval()
    
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            # Validation uses rounding (STE) for accurate BPP estimation
            out_net = model(batch, training_mode="ste")
            out_criterion = criterion(out_net, batch)
            
            loss_meter.update(out_criterion["loss"].item())
            bpp_meter.update(out_criterion["bpp_loss"].item())
            mse_meter.update(out_criterion["mse_loss"].item())
            psnr_meter.update(out_criterion["psnr"].item())
            
            # Visual Log (First batch only)
            if i == 0 and accelerator.is_main_process:
                x_hat = out_net["x_hat"].clamp(0, 1)
                # Compare first 4 images
                n = min(batch.size(0), 4)
                comparison = torch.cat([batch[:n], x_hat[:n]], dim=0)
                grid = make_grid(comparison, nrow=n, padding=2, normalize=False)
                writer.add_image("Val/Reconstruction", grid, epoch)

    if accelerator.is_main_process:
        logger.info(
            f"Test Epoch: [{epoch}] "
            f"Loss: {loss_meter.avg:.4f} | "
            f"Bpp: {bpp_meter.avg:.4f} | "
            f"PSNR: {psnr_meter.avg:.2f} dB"
        )
        writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Val/Bpp", bpp_meter.avg, epoch)
        writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
        
    return loss_meter.avg

# =========================================================
#  MAIN
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="HDMC Training Script")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset root dir")
    parser.add_argument("--save_path", type=str, default="experiments/hdmc_run")
    
    # Training Hyperparams
    parser.add_argument("-e", "--epochs", type=int, default=400)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--aux_lr", dest="aux_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--lmbda", type=float, default=0.0130, help="Rate-distortion parameter")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "ms_ssim"])
    
    # System
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, help="Path to resume checkpoint")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Accelerator Setup
    # Manages Device placement, DDP, Mixed Precision (fp16/bf16)
    accelerator = Accelerator(
        log_with="tensorboard", 
        project_dir=args.save_path,
        mixed_precision="fp16" # Enable mixed precision for Mamba speedup
    )
    set_seed(args.seed)
    
    # 2. Logging Setup
    exp_dir = os.path.join(args.save_path, f"lmbda_{args.lmbda}")
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = None
    writer = None
    if accelerator.is_main_process:
        logger = setup_logger(exp_dir)
        writer = SummaryWriter(os.path.join(exp_dir, "tb_logs"))
        logger.info(f"Config: {vars(args)}")

    # 3. Data Loading
    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    test_transforms = transforms.Compose([
        transforms.CenterCrop(args.patch_size),
        transforms.ToTensor()
    ])

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )

    # 4. Model Initialization
    # N=192, M=320 are the standard "Base" configuration for CompressAI-like models
    model = HDMC(N=192, M=320)
    
    # EMA Model (Shadow copy for validation/inference)
    ema_model = ModelEmaV2(model, decay=0.999)

    # Balancer
    balancer = LossFreeBalancer(update_rate=0.001)

    # 5. Optimizers
    optimizer, aux_optimizer = configure_optimizers(model, args)
    
    # Scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss
    criterion = RateDistortionLoss(lmbda=args.lmbda, loss_type=args.loss_type)

    # 6. Resume
    start_epoch = 0
    best_loss = float("inf")
    
    if args.checkpoint:
        if accelerator.is_main_process:
            logger.info(f"Loading checkpoint: {args.checkpoint}")
        # Map location cpu ensures we don't load onto GPU 0 before Accelerator handles it
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        
        # Load EMA if available
        ema_path = args.checkpoint.replace(".pth.tar", "_ema.pth.tar")
        if os.path.exists(ema_path):
            ema_checkpoint = torch.load(ema_path, map_location="cpu")
            ema_model.module.load_state_dict(ema_checkpoint["state_dict"])
        
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "aux_optimizer" in checkpoint:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        if "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]

    # 7. Prepare via Accelerator
    model, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler
    )
    
    # EMA model needs to be moved to device manually as it's not optimized
    ema_model.module.to(accelerator.device)

    # 8. Training Loop
    global_step = start_epoch * len(train_dataloader)
    
    if accelerator.is_main_process:
        logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        global_step = train_one_epoch(
            model, criterion, train_dataloader, optimizer, aux_optimizer,
            epoch, accelerator, logger, writer, global_step, balancer, ema_model, args
        )
        
        # --- Validation ---
        # Note: We validate using the Main model here for consistency in loss tracking.
        # For production evaluation, you should load the saved EMA weights.
        val_loss = test_epoch(
            epoch, test_dataloader, model, criterion, accelerator, logger, writer
        )
        
        lr_scheduler.step()
        
        # --- Save ---
        if accelerator.is_main_process:
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            
            # Save Main Model
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": accelerator.unwrap_model(model).state_dict(),
                    "loss": val_loss,
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                exp_dir
            )
            
            # Save EMA Model (Weights only usually suffices for inference)
            torch.save(
                {
                    "state_dict": ema_model.module.state_dict(),
                    "epoch": epoch
                },
                os.path.join(exp_dir, "checkpoint_ema.pth.tar")
            )

    if writer:
        writer.close()
    
    if accelerator.is_main_process:
        logger.info("Training complete.")

if __name__ == "__main__":
    main()