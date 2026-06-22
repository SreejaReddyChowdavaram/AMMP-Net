# -*- coding: utf-8 -*-
# losses.py
# Optimization losses and evaluators for AMMP-Net.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MomentumPrototypeBank

# -------------------------------------------------------------------------
# Classification Loss (Cross Entropy with optional Focal variant)
# -------------------------------------------------------------------------
class ClassificationLoss(nn.Module):
    def __init__(self, gamma: float = 0.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.gamma <= 0.0:
            return F.cross_entropy(logits, targets)
            
        log_pt = F.log_softmax(logits, dim=-1)
        pt = torch.exp(log_pt)
        
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        return loss.mean()


# -------------------------------------------------------------------------
# Multi-Level MMD Loss
# -------------------------------------------------------------------------
class MultiLevelMMDLoss(nn.Module):
    def __init__(self, num_kernels: int = 5, kernel_mul: float = 2.0):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_mul = kernel_mul
        
    def _gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_s, n_t = source.size(0), target.size(0)
        total = torch.cat([source, target], dim=0)
        t0 = total.unsqueeze(0).expand(total.size(0), -1, -1)
        t1 = total.unsqueeze(1).expand(-1, total.size(0), -1)
        l2 = ((t0 - t1) ** 2).sum(2)
        
        bandwidth = l2.data.sum() / (total.size(0) ** 2 - total.size(0) + 1e-8)
        bandwidth /= self.kernel_mul ** (self.num_kernels // 2)
        bws = [bandwidth * (self.kernel_mul ** i) for i in range(self.num_kernels)]
        return sum(torch.exp(-l2 / (bw + 1e-8)) for bw in bws)
        
    def mmd(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bs = min(source.size(0), target.size(0))
        if bs == 0:
            return torch.tensor(0.0, device=source.device)
        source, target = source[:bs], target[:bs]
        K = self._gaussian_kernel(source, target)
        XX = K[:bs, :bs]
        YY = K[bs:, bs:]
        XY = K[:bs, bs:]
        YX = K[bs:, :bs]
        return (XX + YY - XY - YX).mean()
        
    def forward(self, fcnn_s, fadam_s, ffusion_s, fcnn_t, fadam_t, ffusion_t):
        mmd_cnn = self.mmd(fcnn_s, fcnn_t)
        mmd_adam = self.mmd(fadam_s, fadam_t)
        mmd_fusion = self.mmd(ffusion_s, ffusion_t)
        return mmd_cnn + mmd_adam + mmd_fusion


# -------------------------------------------------------------------------
# Prototype Alignment Loss
# -------------------------------------------------------------------------
class PrototypeAlignmentLoss(nn.Module):
    def forward(self, feats_s, labels_s, feats_t, labels_t, weights_t, bank: MomentumPrototypeBank) -> torch.Tensor:
        loss = torch.tensor(0.0, device=feats_s.device)
        count = 0
        
        for c in range(bank.num_classes):
            mask_s = (labels_s == c)
            if mask_s.sum() > 0 and bank.tgt_initialized[c]:
                loss += F.mse_loss(feats_s[mask_s], bank.tgt_prototypes[c].unsqueeze(0).expand(mask_s.sum(), -1))
                count += 1
                
        for c in range(bank.num_classes):
            mask_t = (labels_t == c) & (weights_t > 0.5)
            if mask_t.sum() > 0 and bank.src_initialized[c]:
                loss += F.mse_loss(feats_t[mask_t], bank.src_prototypes[c].unsqueeze(0).expand(mask_t.sum(), -1))
                count += 1
                
        if count > 0:
            return loss / count
        return loss


# -------------------------------------------------------------------------
# Prototype Separation Loss
# -------------------------------------------------------------------------
class PrototypeSeparationLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, feats_s, labels_s, feats_t, labels_t, weights_t, bank: MomentumPrototypeBank) -> torch.Tensor:
        loss = torch.tensor(0.0, device=feats_s.device)
        count = 0
        
        for c in range(bank.num_classes):
            mask_s = (labels_s == c)
            if mask_s.sum() > 0:
                for j in range(bank.num_classes):
                    if j != c and bank.tgt_initialized[j]:
                        dist = torch.norm(feats_s[mask_s] - bank.tgt_prototypes[j].unsqueeze(0), p=2, dim=1)
                        loss += torch.clamp(self.margin - dist, min=0.0).pow(2).mean()
                        count += 1
                        
        for c in range(bank.num_classes):
            mask_t = (labels_t == c) & (weights_t > 0.5)
            if mask_t.sum() > 0:
                for j in range(bank.num_classes):
                    if j != c and bank.src_initialized[j]:
                        dist = torch.norm(feats_t[mask_t] - bank.src_prototypes[j].unsqueeze(0), p=2, dim=1)
                        loss += torch.clamp(self.margin - dist, min=0.0).pow(2).mean()
                        count += 1
                        
        if count > 0:
            return loss / count
        return loss


# -------------------------------------------------------------------------
# Reliability-Aware DMSL
# -------------------------------------------------------------------------
class ReliabilityAwareDMSL(nn.Module):
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=-1)
        kl = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1) * (self.temperature ** 2)
        return (kl * weights).mean()


# -------------------------------------------------------------------------
# Prediction Reliability and Entropy Filtering Utilities
# -------------------------------------------------------------------------
def compute_reliability_weights(logits: torch.Tensor) -> tuple:
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
    
    C = logits.size(-1)
    max_entropy = math.log(C) if C > 1 else 1.0
    norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)
    
    weights = torch.sigmoid(5.0 * (0.5 - norm_entropy))
    pseudo = probs.argmax(dim=-1)
    return probs, weights, pseudo, norm_entropy


def sharpen(probs: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    if temperature >= 1.0:
        return probs
    sharp = probs.pow(1.0 / temperature)
    return sharp / sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)


# -------------------------------------------------------------------------
# Combined AMMP-Net Loss Evaluator
# -------------------------------------------------------------------------
class AMMPNetLossEvaluator(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, cfg: dict):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        self.cls_loss_fn = ClassificationLoss(gamma=2.0)
        self.mgda_loss_fn = MultiLevelMMDLoss()
        self.align_loss_fn = PrototypeAlignmentLoss()
        self.sep_loss_fn = PrototypeSeparationLoss(margin=1.0)
        self.dmsl_loss_fn = ReliabilityAwareDMSL(temperature=2.0)
        
        self.lambda_cls = cfg.get("lambda_cls", 1.0)
        self.lambda_mgda = cfg.get("lambda_mgda", 0.2)
        self.lambda_align = cfg.get("lambda_align", 0.5)
        self.lambda_sep = cfg.get("lambda_sep", 0.05)
        self.lambda_dmsl = cfg.get("lambda_dmsl", 1.0)
        
    def forward(self, logits_s, y_s,
                fcnn_s, fadam_s, ffusion_s,
                logits_t, fcnn_t, fadam_t, ffusion_t,
                logits_t_teach, ffusion_t_teach,
                prototype_bank,
                use_mgda: bool = True,
                use_mpam: bool = True,
                use_sep: bool = True,
                use_dmsl: bool = True) -> tuple:
        
        l_cls = self.cls_loss_fn(logits_s, y_s)
        
        l_mgda = torch.tensor(0.0, device=logits_s.device)
        if use_mgda:
            l_mgda = self.mgda_loss_fn(fcnn_s, fadam_s, ffusion_s, fcnn_t, fadam_t, ffusion_t)
            
        _, w_t, pseudo_t, _ = compute_reliability_weights(logits_t_teach)
        
        if use_mpam:
            prototype_bank.update(ffusion_s, y_s, domain="source")
            high_rel_mask = (w_t > 0.5)
            if high_rel_mask.any():
                prototype_bank.update(ffusion_t[high_rel_mask], pseudo_t[high_rel_mask], domain="target")
                
        l_align = torch.tensor(0.0, device=logits_s.device)
        l_sep = torch.tensor(0.0, device=logits_s.device)
        if use_mpam:
            l_align = self.align_loss_fn(ffusion_s, y_s, ffusion_t, pseudo_t, w_t, prototype_bank)
            if use_sep:
                l_sep = self.sep_loss_fn(ffusion_s, y_s, ffusion_t, pseudo_t, w_t, prototype_bank)
                
        l_dmsl = torch.tensor(0.0, device=logits_s.device)
        if use_dmsl:
            l_dmsl = self.dmsl_loss_fn(logits_t, logits_t_teach, weights=w_t)
            
        loss = (
            self.lambda_cls * l_cls +
            self.lambda_mgda * l_mgda +
            self.lambda_align * l_align +
            self.lambda_dmsl * l_dmsl
        )
        
        loss_dict = {
            "loss": loss.item(),
            "l_cls": l_cls.item(),
            "l_mgda": l_mgda.item(),
            "l_align": l_align.item(),
            "l_sep": l_sep.item(),
            "l_dmsl": l_dmsl.item()
        }
        
        return loss, loss_dict
