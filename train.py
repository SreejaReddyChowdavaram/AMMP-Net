# -*- coding: utf-8 -*-
# train.py
# Main execution script, configurations, and trainer for AMMP-Net.

import os
import time
import math
import argparse
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import dataset, models, and utils components
from dataset import get_data_loaders
from models import FullAMMPNet, SimpleAMMPModel
from losses import AMMPNetLossEvaluator
from utils import (
    seed_everything,
    parameter_counter,
    save_config,
    save_predictions,
    save_metrics,
    evaluate_model
)

# -------------------------------------------------------------------------
# Configuration Settings (Merged from config.py)
# -------------------------------------------------------------------------
_DEFAULT_CFG = {
    "patch_size": 11,
    "batch_size": 64,
    "learning_rate": 0.0003,
    "epochs": 50,
    "seed": 42,
    "patience": 10,
    "grad_clip_norm": 1.0,
    "warmup_epochs": 5,
    "ema_momentum": 0.99,
    "memory_bank_momentum": 0.9,
    
    # Loss scaling weights
    "lambda_cls": 1.0,
    "lambda_mgda": 0.05,
    "lambda_align": 0.05,
    "lambda_sep": 0.01,
    "lambda_dmsl": 0.05,
    
    # Dataset splits and loading
    "spatial_split": False,
    "split_ratio": 0.5,
    "train_ratio": 1.0,
    "max_target_samples": 1000,
    "test_batch_size": 256,
    "verbose": False,
    "fast_mode": True,
}

_DATASET_OVERRIDES = {
    "houston": {
        "dataset_name": "Houston",
        "in_channels": 48,
        "num_classes": 7,
        "patch_size": 11,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "epochs": 50,
    },
    "hyrank": {
        "dataset_name": "HyRANK",
        "in_channels": 176,
        "num_classes": 12,
        "patch_size": 7,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "epochs": 40,
    },
    "mff": {
        "dataset_name": "MFF",
        "in_channels": 64,
        "num_classes": 12,
        "patch_size": 11,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "epochs": 50,
    }
}

def get_config(dataset_name: str) -> dict:
    """
    Returns a unified configuration dictionary for the specified dataset.
    """
    key = dataset_name.lower().strip()
    cfg = dict(_DEFAULT_CFG)
    
    if key in _DATASET_OVERRIDES:
        cfg.update(_DATASET_OVERRIDES[key])
    else:
        cfg["dataset_name"] = dataset_name.capitalize()

    # Define source and target paths relative to workspace datasets folder
    base_dir = Path(__file__).resolve().parent / "datasets"
    
    if key == "houston":
        cfg["source_path"] = str(base_dir / "Houston" / "Houston13.mat")
        cfg["source_gt"] = str(base_dir / "Houston" / "Houston13_7gt.mat")
        cfg["target_path"] = str(base_dir / "Houston" / "Houston18.mat")
        cfg["target_gt"] = str(base_dir / "Houston" / "Houston18_7gt.mat")
    elif key == "hyrank":
        cfg["source_path"] = str(base_dir / "HyRANK" / "Dioni.mat")
        cfg["source_gt"] = str(base_dir / "HyRANK" / "Dioni_gt.mat")
        cfg["target_path"] = str(base_dir / "HyRANK" / "Loukia.mat")
        cfg["target_gt"] = str(base_dir / "HyRANK" / "Loukia_gt.mat")
    elif key == "mff":
        cfg["source_path"] = str(base_dir / "MFF" / "MFF_SD.mat")
        cfg["source_gt"] = str(base_dir / "MFF" / "MFF_SD_gt.mat")
        cfg["target_path"] = str(base_dir / "MFF" / "MFF_TD.mat")
        cfg["target_gt"] = str(base_dir / "MFF" / "MFF_TD_gt.mat")
    else:
        # Fallback paths for custom/unrecognized datasets
        custom_base = base_dir / cfg["dataset_name"]
        cfg["source_path"] = str(custom_base / f"{key}_src.mat")
        cfg["source_gt"] = str(custom_base / f"{key}_src_gt.mat")
        cfg["target_path"] = str(custom_base / f"{key}_tgt.mat")
        cfg["target_gt"] = str(custom_base / f"{key}_tgt_gt.mat")
        
    return cfg

def get_supported_datasets() -> list:
    """
    Returns list of supported datasets.
    """
    return list(_DATASET_OVERRIDES.keys())


# -------------------------------------------------------------------------
# AMMP Trainer Class (Merged from trainer.py)
# -------------------------------------------------------------------------
class AMMPTrainer:
    """
    Coordinates training epochs, validation checks, and checkpoint saving.
    """
    def __init__(self, model, src_loader, tgt_train_loader, tgt_test_loader,
                 cfg: dict, device: torch.device, use_mgda: bool = True,
                 use_mpam: bool = True, use_sep: bool = True, use_dmsl: bool = True):
        self.model = model
        self.src_loader = src_loader
        self.tgt_train_loader = tgt_train_loader
        self.tgt_test_loader = tgt_test_loader
        self.cfg = cfg
        self.device = device
        
        if self.cfg.get("fast_mode", True):
            self.use_mgda = False
            self.use_mpam = False
            self.use_sep = False
            self.use_dmsl = False
        else:
            self.use_mgda = use_mgda
            self.use_mpam = use_mpam
            self.use_sep = use_sep
            self.use_dmsl = use_dmsl
        
        # Setup loss evaluator
        self.loss_evaluator = AMMPNetLossEvaluator(
            num_classes=cfg["num_classes"],
            feature_dim=model.student.d_model,
            cfg=cfg
        ).to(device)
        
        # Optimizer & Scheduler
        self.lr = cfg["learning_rate"]
        self.optimizer = torch.optim.Adam(
            self.model.student.parameters(),
            lr=self.lr,
            weight_decay=1e-4
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cfg["epochs"],
            eta_min=self.lr * 0.01
        )
        
        # Best checkpoint tracking
        self.best_oa = -1.0
        self.best_epoch = 0
        self.best_metrics = {}
        
        # Mixed precision GradScaler
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.device.type == 'cuda')
        )
        
        # Setup fast validation loader to decrease epoch loading/evaluation time
        dataset_test = self.tgt_test_loader.dataset
        num_test_samples = len(dataset_test)
        if num_test_samples > 1000:
            from torch.utils.data import Subset
            rng = np.random.default_rng(cfg.get("seed", 42))
            fast_indices = rng.choice(num_test_samples, 1000, replace=False).tolist()
            fast_ds = Subset(dataset_test, fast_indices)
            self.tgt_test_fast_loader = torch.utils.data.DataLoader(
                fast_ds,
                batch_size=cfg.get("test_batch_size", 256),
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2
            )
        else:
            self.tgt_test_fast_loader = self.tgt_test_loader

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        
        src_iter = iter(self.src_loader)
        fast_mode = self.cfg.get("fast_mode", True)
        
        if fast_mode:
            n_batches = len(self.src_loader)
        else:
            tgt_iter = iter(self.tgt_train_loader)
            n_batches = min(len(self.src_loader), len(self.tgt_train_loader))
        
        total_loss = 0.0
        losses_sum = {
            "l_cls": 0.0,
            "l_mgda": 0.0,
            "l_align": 0.0,
            "l_sep": 0.0,
            "l_dmsl": 0.0
        }
        
        for batch_idx in range(n_batches):
            try:
                xs, ys = next(src_iter)
            except StopIteration:
                src_iter = iter(self.src_loader)
                xs, ys = next(src_iter)
                
            self.optimizer.zero_grad(set_to_none=True)
            
            if fast_mode:
                xs, ys = xs.to(self.device), ys.to(self.device)
                with torch.cuda.amp.autocast(
                    dtype=torch.bfloat16,
                    enabled=(self.device.type == 'cuda')
                ):
                    logits_s, fcnn_s, fadam_s, ffusion_s = self.model.student(xs)
                    loss = self.loss_evaluator.cls_loss_fn(logits_s, ys)
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.student.parameters(),
                    self.cfg.get("grad_clip_norm", 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                loss_dict = {
                    "loss": loss.item(),
                    "l_cls": loss.item(),
                    "l_mgda": 0.0,
                    "l_align": 0.0,
                    "l_sep": 0.0,
                    "l_dmsl": 0.0
                }
            else:
                try:
                    xt, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(self.tgt_train_loader)
                    xt, _ = next(tgt_iter)
                    
                xs, ys, xt = xs.to(self.device), ys.to(self.device), xt.to(self.device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(self.device.type == 'cuda')):
                    outputs = self.model(xs, ys, xt)
                    (
                        logits_s, fcnn_s, fadam_s, ffusion_s,
                        logits_t, fcnn_t, fadam_t, ffusion_t,
                        logits_t_teach, ffusion_t_teach
                    ) = outputs
                    
                    tau = min(0.45, 0.15 + 0.30*(epoch/self.cfg["epochs"]))
                    
                    loss, loss_dict = self.loss_evaluator(
                        logits_s, ys,
                        fcnn_s, fadam_s, ffusion_s,
                        logits_t, fcnn_t, fadam_t, ffusion_t,
                        logits_t_teach, ffusion_t_teach,
                        self.model.prototype_bank,
                        use_mgda=self.use_mgda,
                        use_mpam=self.use_mpam,
                        use_sep=self.use_sep,
                        use_dmsl=self.use_dmsl,
                        tau=tau
                    )
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.student.parameters(),
                    self.cfg.get("grad_clip_norm", 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                global_step = (epoch - 1) * n_batches + batch_idx
                total_steps = self.cfg["epochs"] * n_batches
                momentum = 0.99 + 0.009 * (global_step / total_steps)
                momentum = min(momentum, 0.999)
                self.model.update_teacher(momentum)
                
            total_loss += loss.item()
            for k in losses_sum:
                losses_sum[k] += loss_dict[k]
                
        avg_loss = total_loss / n_batches
        avg_losses = {k: v / n_batches for k, v in losses_sum.items()}
        avg_losses["loss"] = avg_loss
        
        return avg_losses

    def validate(self, fast: bool = True) -> dict:
        loader = self.tgt_test_fast_loader if fast else self.tgt_test_loader
        metrics = evaluate_model(self.model.student, loader, self.device)
        return metrics

    def run_training(self, save_dir: str, log_csv_path: str) -> dict:
        os.makedirs(save_dir, exist_ok=True)
        
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "oa", "aa", "kappa"])
            
        if self.cfg.get("verbose", False):
            print(f"[Trainer] Starting training loop for {self.cfg['epochs']} epochs.")
            print(f"[Trainer] Output folder: {save_dir}")
            print("-" * 60)
        
        val_metrics = {"oa": 0.0, "aa": 0.0, "kappa": 0.0}
        for epoch in range(1, self.cfg["epochs"] + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            self.scheduler.step()
            
            if epoch % 5 == 0 or epoch == self.cfg["epochs"]:
                val_metrics = self.validate(fast=True)
                
            elapsed = time.time() - t0
            
            print(
                f"Epoch {epoch:03d}/{self.cfg['epochs']} | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"OA: {val_metrics['oa']:.2f} | "
                f"Time: {elapsed:.2f}s"
            )
            
            with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_metrics['loss']:.4f}",
                    f"{val_metrics['oa']:.4f}",
                    f"{val_metrics['aa']:.4f}",
                    f"{val_metrics['kappa']*100:.4f}"
                ])
                
            if epoch % 5 == 0 or epoch == self.cfg["epochs"]:
                if val_metrics["oa"] > self.best_oa:
                    self.best_oa = val_metrics["oa"]
                    self.best_epoch = epoch
                    self.best_metrics = val_metrics
                    
                    torch.save(
                        self.model.student.state_dict(),
                        os.path.join(save_dir, "best_model.pt")
                    )
                    if self.cfg.get("verbose", False):
                        print(f" ==> Saved new best model at epoch {epoch} with OA: {val_metrics['oa']:.2f}% (fast subset)")
                    
        if self.cfg.get("verbose", False):
            print("-" * 60)
            print(f"[Trainer] Training completed. Best Epoch: {self.best_epoch} with OA: {self.best_oa:.2f}% (fast subset)")
        return self.best_metrics


# -------------------------------------------------------------------------
# Main Execution (Merged from train_new.py)
# -------------------------------------------------------------------------
def parse_args():
    supported = get_supported_datasets()
    parser = argparse.ArgumentParser(
        description="AMMP-Net Cross-Scene Hyperspectral Image Classification Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help=f"Target dataset pair key. Supported: {supported}")
    
    # Optional hyperparameter overrides
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs to train (overrides default)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Mini-batch size (overrides default)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Initial learning rate (overrides default)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Model feature projection dimension")
    
    # Ablation and loss options (1 = True, 0 = False)
    parser.add_argument("--use_mgda", type=int, default=1, choices=[0, 1],
                        help="Use Multi-Level MMD Loss (MGDA) for adaptation")
    parser.add_argument("--use_mpam", type=int, default=1, choices=[0, 1],
                        help="Use Momentum Prototype Alignment Module / Loss")
    parser.add_argument("--use_sep", type=int, default=1, choices=[0, 1],
                        help="Use Prototype Separation Loss in addition to alignment")
    parser.add_argument("--use_dmsl", type=int, default=1, choices=[0, 1],
                        help="Use Reliability-Aware Dynamic Mutual Self-Distillation Loss")
    
    # Mode shortcut
    parser.add_argument("--backbone_only", action="store_true",
                        help="Run standard CNN backbone classifier training on source (ignores target adaptation)")
    
    # Verbose and fast mode options
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1],
                        help="Enable verbose output logging")
    parser.add_argument("--fast_mode", type=int, default=1, choices=[0, 1],
                        help="Enable fast training mode (mixed precision, backbone only, reduced target size)")
                        
    return parser.parse_args()


def main():
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    
    # 1. Reproducibility
    seed_everything(args.seed)
    
    # 2. Configuration Settings
    cfg = get_config(args.dataset)
    
    # Override defaults if specified in command-line arguments
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["learning_rate"] = args.lr
    if args.fast_mode is not None:
        cfg["fast_mode"] = (args.fast_mode == 1)
    if args.verbose is not None:
        cfg["verbose"] = (args.verbose == 1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine adaptation settings
    if cfg.get("fast_mode", True):
        use_mgda = False
        use_mpam = False
        use_sep = False
        use_dmsl = False
        mode_name = "Backbone-only"
    elif args.backbone_only:
        use_mgda = False
        use_mpam = False
        use_sep = False
        use_dmsl = False
        mode_name = "Backbone-only"
    else:
        use_mgda = (args.use_mgda == 1)
        use_mpam = (args.use_mpam == 1)
        use_sep = (args.use_sep == 1)
        use_dmsl = (args.use_dmsl == 1)
        if not (use_mgda or use_mpam or use_dmsl):
            mode_name = "Backbone-only"
        else:
            mode_name = "Full AMMP-Net"
            
    # 3. Load Dataset loaders
    src_loader, tgt_train_loader, tgt_test_loader, Ys_train, dinfo, cfg = get_data_loaders(
        cfg=cfg,
        train_ratio=cfg.get("train_ratio", 0.05),
        seed=args.seed,
        max_target_samples=cfg.get("max_target_samples", 1000),
        cache_dataset=True
    )
    
    in_channels = dinfo["in_channels"]
    num_classes = dinfo["num_classes"]
    
    # 4. Instantiate Model
    if cfg.get("verbose", False):
        print(f"Model Dimension : {args.d_model}")
    model = FullAMMPNet(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=cfg["patch_size"],
        d_model=args.d_model,
        d_state=16,
        memory_bank_momentum=cfg.get("memory_bank_momentum", 0.9)
    )
        
    model = model.to(device)
    n_params = parameter_counter(model.student)
    
    # 5. Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = cfg["dataset_name"]
    run_dir = os.path.join("results", dataset_name, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    log_csv_path = os.path.join(run_dir, "training_log.csv")
    save_config(cfg, run_dir)
    
    # 6. Initialize trainer
    trainer = AMMPTrainer(
        model=model,
        src_loader=src_loader,
        tgt_train_loader=tgt_train_loader,
        tgt_test_loader=tgt_test_loader,
        cfg=cfg,
        device=device,
        use_mgda=use_mgda,
        use_mpam=use_mpam,
        use_sep=use_sep,
        use_dmsl=use_dmsl
    )
    
    # 7. Execute Training Loop
    t_start = time.time()
    best_metrics = trainer.run_training(run_dir, log_csv_path)
    t_duration = time.time() - t_start
    
    # 8. Load best model weights and verify final metrics
    best_weights_path = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(best_weights_path):
        model.student.load_state_dict(torch.load(best_weights_path, map_location=device))
        
    final_metrics = trainer.validate(fast=False)
    final_metrics["best_epoch"] = trainer.best_epoch
    final_metrics["model_parameters"] = n_params
    
    save_metrics(final_metrics, run_dir, dataset_name, cfg, dinfo)
    save_predictions(final_metrics["predictions"], run_dir)
    np.save(os.path.join(run_dir, "confusion_matrix.npy"), final_metrics["confusion_matrix"])
    
    # 9. Format console output precisely
    if cfg.get("verbose", False):
        print("\n" + "=" * 60)
        print("  AMMP-NET EXPERIMENTAL RUN COMPLETE")
        print("=" * 60)
        print(f"  Dataset         : {dataset_name}")
        print(f"  Parameters      : {n_params}")
        print(f"  Training Samples: {dinfo['num_source_train']}")
        print(f"  Testing Samples : {dinfo['num_target_test']}")
        print(f"  Input Shape     : (B, {in_channels}, {cfg['patch_size']}, {cfg['patch_size']})")
        print(f"  OA              : {final_metrics['oa']:.2f}%")
        print(f"  AA              : {final_metrics['aa']:.2f}%")
        print(f"  Kappa           : {final_metrics['kappa']*100:.2f}%")
        print(f"  Training Time   : {t_duration:.2f}s")
        print(f"  Best Epoch      : {trainer.best_epoch}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
