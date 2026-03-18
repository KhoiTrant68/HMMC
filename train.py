import argparse
import logging
import math
import os
import shutil
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from compressai.datasets import ImageFolder
from timm.utils import ModelEmaV2
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from loss.loss import AverageMeter, RateDistortionLoss
from model.hmmc import HMMC


class LossFreeBalancer:
    def __init__(self, num_experts=4, update_rate=0.001):
        self.num_experts = num_experts
        self.update_rate = update_rate

    @torch.no_grad()
    def update_biases(self, model, router_data_tuple, accelerator):
        """
        Prevents MoE biases from drifting apart across different GPUs.
        """
        if not router_data_tuple:
            return

        for i, layer_data in enumerate(router_data_tuple):
            if not (isinstance(layer_data, (tuple, list)) and len(layer_data) == 2):
                continue

            _, topk_indices = layer_data
            local_counts = (
                torch.bincount(topk_indices.flatten(), minlength=self.num_experts)
                .float()
                .to(accelerator.device)
            )

            # Synchronize local bincounts across all ranks
            if accelerator.num_processes > 1:
                global_counts = accelerator.reduce(local_counts, reduction="sum")
            else:
                global_counts = local_counts

            avg_count = global_counts.mean()
            error = avg_count - global_counts
            error_ratio = error / (avg_count + 1e-8)
            update_step = self.update_rate * error_ratio.clamp(-5.0, 5.0)

            moe_module = self._get_module_by_index(model, i)
            if moe_module is not None and hasattr(moe_module, "expert_biases"):
                moe_module.expert_biases.data.add_(update_step)

    def _get_module_by_index(self, model, index):
        if not hasattr(model, "dt_cross_attention"):
            return None
        num_standard = len(model.dt_cross_attention)
        if index < num_standard:
            return model.dt_cross_attention[index]
        elif index == num_standard:
            return model.moe_anchor
        elif index == num_standard + 1:
            return model.moe_non_anchor
        return None


def setup_logger(log_dir):
    logger = logging.getLogger("Train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def configure_optimizers(net, args):
    params_dict = dict(net.named_parameters())
    params = {n: p for n, p in params_dict.items() if not n.endswith(".quantiles")}
    aux_params = {n: p for n, p in params_dict.items() if n.endswith(".quantiles")}

    optimizer = optim.AdamW(
        [p for p in params.values() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    aux_optimizer = optim.Adam(
        [p for p in aux_params.values() if p.requires_grad], lr=args.aux_learning_rate
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model,
    criterion,
    train_dataloader,
    optimizer,
    aux_optimizer,
    epoch,
    clip_max_norm,
    accelerator,
    logger,
    writer,
    global_step,
    balancer,
    ema_model,
    training_mode,
    args,
):
    model.train()
    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    dist_meter = AverageMeter()
    bias_stats_logged = False

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(batch, training_mode=training_mode)
        out_criterion = criterion(out_net, batch)
        loss = out_criterion["loss"]

        unwrapped_model = accelerator.unwrap_model(model)
        ortho_loss = 0.0
        dict_count = 0
        for module in unwrapped_model.modules():
            if hasattr(module, "experts_high") and isinstance(
                module.experts_high, nn.Parameter
            ):
                W = module.experts_high
                W_norm = F.normalize(W, p=2, dim=-1)
                sim_matrix = torch.matmul(W_norm, W_norm.transpose(0, 1))
                eye = torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
                ortho_loss += F.mse_loss(sim_matrix, eye)
                dict_count += 1

        if dict_count > 0:
            loss = loss + (0.01 * (ortho_loss / dict_count))

        accelerator.backward(loss)
        if clip_max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # Update biases natively utilizing DDP accelerator synchronization
        if "router_logits" in out_net and out_net["router_logits"]:
            balancer.update_biases(
                unwrapped_model, out_net["router_logits"], accelerator
            )

        aux_loss = unwrapped_model.aux_loss()
        accelerator.backward(aux_loss)
        aux_optimizer.step()

        if ema_model is not None:
            ema_model.update(unwrapped_model)

        loss_meter.update(loss.item())
        bpp_meter.update(out_criterion["bpp_loss"].item())
        dist_meter.update(out_criterion["mse_loss"].item())

        if i % args.print_freq == 0 and accelerator.is_main_process:
            logger.info(
                f"Epoch [{epoch}][{i}/{len(train_dataloader)}] Loss: {loss_meter.val:.4f} | Bpp: {bpp_meter.val:.4f} | MSE: {dist_meter.val:.6f}"
            )
            writer.add_scalar("Train/Loss", loss_meter.val, global_step)
            writer.add_scalar("Train/Bpp", bpp_meter.val, global_step)
            writer.add_scalar(
                "Train/MoE_Imbalance",
                out_criterion.get("moe_imbalance", 0.0),
                global_step,
            )

            if not bias_stats_logged and hasattr(unwrapped_model, "dt_cross_attention"):
                if hasattr(unwrapped_model.dt_cross_attention[0], "expert_biases"):
                    biases = unwrapped_model.dt_cross_attention[0].expert_biases
                    writer.add_scalar(
                        "Debug/Bias_Max", biases.max().item(), global_step
                    )
                    writer.add_scalar(
                        "Debug/Bias_Min", biases.min().item(), global_step
                    )
                    bias_stats_logged = True
        global_step += 1
    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, accelerator, logger, writer):
    model.eval()
    loss_meter, bpp_meter, psnr_meter = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            out_net = model(batch, training_mode="ste")
            out_criterion = criterion(out_net, batch)

            loss_meter.update(out_criterion["loss"].item())
            bpp_meter.update(out_criterion["bpp_loss"].item())
            psnr_meter.update(out_criterion["psnr"].item())

            if i == 0 and accelerator.is_main_process:
                x_hat = out_net["x_hat"].clamp(0, 1)
                n = min(batch.size(0), 4)
                grid = make_grid(
                    torch.cat([batch[:n], x_hat[:n]], dim=0), nrow=n, padding=2
                )
                writer.add_image("Val/Reconstruction", grid, epoch)

    if accelerator.is_main_process:
        logger.info(
            f"Test Epoch [{epoch}] Loss: {loss_meter.avg:.4f} | Bpp: {bpp_meter.avg:.4f} | PSNR: {psnr_meter.avg:.2f} dB"
        )
        writer.add_scalar("Val/Loss", loss_meter.avg, epoch)
        writer.add_scalar("Val/PSNR", psnr_meter.avg, epoch)
    return loss_meter.avg


def save_checkpoint(state, is_best, save_path, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, filename),
            os.path.join(save_path, "checkpoint_best.pth.tar"),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("-e", "--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--aux-lr", dest="aux_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lmbda", type=float, default=0.0130)
    parser.add_argument("--clip_max_norm", type=float, default=1.0)
    parser.add_argument("--update_rate", type=float, default=0.001)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1926)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.save_path)
    save_path = os.path.join(args.save_path, f"lambda_{args.lmbda}")
    os.makedirs(save_path, exist_ok=True)

    logger, writer = None, None
    if accelerator.is_main_process:
        logger = setup_logger(save_path)
        writer = SummaryWriter(os.path.join(save_path, "tb"))

    train_dataset = ImageFolder(
        args.dataset,
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomCrop(args.patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    test_dataset = ImageFolder(
        args.dataset,
        split="valid",
        transform=transforms.Compose(
            [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
        ),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    net = HMMC(N=192, M=320)
    ema_model = ModelEmaV2(net, decay=0.999)
    balancer = LossFreeBalancer(num_experts=4, update_rate=args.update_rate)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    start_epoch = 0
    best_loss = float("inf")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        net.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["scheduler"])
        best_loss = checkpoint.get("loss", best_loss)

        ema_ckpt_path = args.checkpoint.replace(
            "checkpoint.pth.tar", "checkpoint_ema.pth.tar"
        )
        if os.path.exists(ema_ckpt_path):
            ema_ckpt = torch.load(ema_ckpt_path, map_location="cpu")
            ema_model.module.load_state_dict(ema_ckpt["state_dict"])
            if accelerator.is_main_process:
                logger.info("Successfully loaded EMA weights.")

    net, optimizer, aux_optimizer, train_dataloader, test_dataloader, lr_scheduler = (
        accelerator.prepare(
            net,
            optimizer,
            aux_optimizer,
            train_dataloader,
            test_dataloader,
            lr_scheduler,
        )
    )
    ema_model.module.to(accelerator.device)

    global_step = start_epoch * len(train_dataloader)

    for epoch in range(start_epoch, args.epochs):
        training_mode = "noise" if epoch < args.epochs * 0.75 else "ste"
        if epoch == int(args.epochs * 0.75):
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate * 0.1

        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            accelerator,
            logger,
            writer,
            global_step,
            balancer,
            ema_model,
            training_mode,
            args,
        )
        loss = test_epoch(
            epoch, test_dataloader, net, criterion, accelerator, logger, writer
        )
        lr_scheduler.step()

        if accelerator.is_main_process:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": accelerator.unwrap_model(net).state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                save_path,
            )
            torch.save(
                {"state_dict": ema_model.module.state_dict(), "epoch": epoch},
                os.path.join(save_path, "checkpoint_ema.pth.tar"),
            )

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
