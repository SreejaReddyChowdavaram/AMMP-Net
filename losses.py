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
        
    def forward(self, fcnn_s, fadam_s, ffusion_s,
            fcnn_t, fadam_t, ffusion_t):

        mmd_cnn = self.mmd(fcnn_s, fcnn_t)
        mmd_adam = self.mmd(fadam_s, fadam_t)
        mmd_fusion = self.mmd(ffusion_s, ffusion_t)

        return (
            0.1 * mmd_cnn +
            0.3 * mmd_adam +
            1.0 * mmd_fusion
        )

# -------------------------------------------------------------------------
# Prototype Alignment Loss
# -------------------------------------------------------------------------
class PrototypeAlignmentLoss(nn.Module):
    def forward(self, feats_s, labels_s, feats_t, labels_t, weights_t, bank: MomentumPrototypeBank) -> torch.Tensor:
        loss = torch.tensor(0.0, device=feats_s.device)
        count = 0
        for c in range(bank.num_classes):
            mask_s = (labels_s == c)
            mask_t = (labels_t == c) & (weights_t > 0.5)
            
            if mask_s.sum() > 0:
                src_proto = feats_s[mask_s].mean(dim=0)
            elif bank.src_initialized[c]:
                src_proto = bank.src_prototypes[c]
            else:
                continue
                
            if mask_t.sum() > 0:
                denom = weights_t[mask_t].sum()
                if denom > 1e-5:
                    tgt_proto = (feats_t * weights_t.unsqueeze(1))[mask_t].sum(dim=0) / denom
                else:
                    tgt_proto = bank.tgt_prototypes[c]
            elif bank.tgt_initialized[c]:
                tgt_proto = bank.tgt_prototypes[c]
            else:
                continue
                
            # L2 normalise to project features onto a unit hypersphere (cosine metric alignment)
            src_proto = F.normalize(src_proto, p=2, dim=0)
            tgt_proto = F.normalize(tgt_proto, p=2, dim=0)
            loss += torch.norm(src_proto - tgt_proto, p=2) ** 2
            count += 1
            
        if count > 0:
            return loss / count
        return loss

class PrototypeSeparationLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, feats_s, labels_s, feats_t, labels_t, weights_t, bank: MomentumPrototypeBank) -> torch.Tensor:
        loss = torch.tensor(0.0, device=feats_s.device)
        count = 0
        
        # Normalize representations and bank prototypes
        src_protos_norm = F.normalize(bank.src_prototypes, p=2, dim=1)
        tgt_protos_norm = F.normalize(bank.tgt_prototypes, p=2, dim=1)
        feats_s_norm = F.normalize(feats_s, p=2, dim=1)
        feats_t_norm = F.normalize(feats_t, p=2, dim=1)
        
        for c in range(bank.num_classes):
            mask_s = (labels_s == c)
            if mask_s.sum() > 0:
                for j in range(bank.num_classes):
                    if j != c and bank.tgt_initialized[j]:
                        dist = torch.norm(feats_s_norm[mask_s] - tgt_protos_norm[j].unsqueeze(0), p=2, dim=1)
                        loss += torch.clamp(self.margin - dist, min=0.0).pow(2).mean()
                        count += 1
                        
        for c in range(bank.num_classes):
            mask_t = (labels_t == c) & (weights_t > 0.5)
            if mask_t.sum() > 0:
                for j in range(bank.num_classes):
                    if j != c and bank.src_initialized[j]:
                        dist = torch.norm(feats_t_norm[mask_t] - src_protos_norm[j].unsqueeze(0), p=2, dim=1)
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
        # Pseudo labels from teacher predictions
        pseudo = teacher_logits.argmax(dim=1)
        ce_loss = F.cross_entropy(student_logits, pseudo, reduction="none")
        weighted_ce = (ce_loss * weights).mean()
        
        # Softmax Probability KL distillation for stable gradients
        p_s = F.log_softmax(student_logits / self.temperature, dim=-1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=-1)
        consistency = F.kl_div(p_s, p_t, reduction="batchmean") * (self.temperature ** 2)
        return weighted_ce + consistency


# -------------------------------------------------------------------------
# Prediction Reliability and Entropy Filtering Utilities
# -------------------------------------------------------------------------
def compute_reliability_weights(logits: torch.Tensor, tau: float = 0.3) -> tuple:
    """Compute reliability weights based on confidence.
    w_i = sigmoid(beta * (p_i - tau)) where p_i = max class probability.
    """
    probs = F.softmax(logits, dim=-1)
    confidence, _ = probs.max(dim=-1)
    beta = 20.0
    weights = torch.sigmoid(beta * (confidence - tau))
    pseudo = probs.argmax(dim=-1)
    return probs, weights, pseudo, confidence


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

        self.cls_loss_fn = ClassificationLoss(gamma=0.0)
        self.mgda_loss_fn = MultiLevelMMDLoss()
        self.align_loss_fn = PrototypeAlignmentLoss()
        self.sep_loss_fn = PrototypeSeparationLoss(margin=1.0)
        self.dmsl_loss_fn = ReliabilityAwareDMSL(temperature=2.0)

        self.lambda_cls = cfg.get("lambda_cls", 1.0)
        self.lambda_mgda = cfg.get("lambda_mgda", 0.2)
        self.lambda_align = cfg.get("lambda_align", 0.5)
        self.lambda_sep = cfg.get("lambda_sep", 0.05)
        self.lambda_dmsl = cfg.get("lambda_dmsl", 1.0)

    def forward(
        self,
        logits_s,
        y_s,
        fcnn_s,
        fadam_s,
        ffusion_s,
        logits_t,
        fcnn_t,
        fadam_t,
        ffusion_t,
        logits_t_teach,
        ffusion_t_teach,
        prototype_bank,
        use_mgda=True,
        use_mpam=True,
        use_sep=True,
        use_dmsl=True,
        tau=0.7,
    ):

        # -----------------------------
        # Classification Loss
        # -----------------------------
        l_cls = self.cls_loss_fn(logits_s, y_s)

        # -----------------------------
        # MMD Loss
        # -----------------------------
        l_mgda = torch.tensor(0.0, device=logits_s.device)

        if use_mgda:
            l_mgda = self.mgda_loss_fn(
                fcnn_s,
                fadam_s,
                ffusion_s,
                fcnn_t,
                fadam_t,
                ffusion_t,
            )

        # -----------------------------
        # Reliability Estimation
        # -----------------------------
        if use_mpam or use_dmsl:

            _, w_t, pseudo_t, confidence = compute_reliability_weights(
                logits_t_teach,
                tau=tau,
            )

            pass

        else:

            batch_size = logits_t.shape[0]

            w_t = torch.zeros(
                batch_size,
                device=logits_t.device,
            )

            pseudo_t = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=logits_t.device,
            )

            confidence = torch.zeros(
                batch_size,
                device=logits_t.device,
            )

        # -----------------------------
        # Prototype Alignment
        # -----------------------------
        l_align = torch.tensor(0.0, device=logits_s.device)
        l_sep = torch.tensor(0.0, device=logits_s.device)

        if use_mpam:

            prototype_bank.update(
                ffusion_s,
                y_s,
                domain="source",
            )

            k = max(1, int(0.3 * confidence.shape[0]))

            _, idx = torch.topk(confidence, k)

            high_rel_mask = torch.zeros_like(
                confidence,
                dtype=torch.bool,
            )

            high_rel_mask[idx] = True

            pass

            if high_rel_mask.any():

                prototype_bank.update(
                    ffusion_t[high_rel_mask],
                    pseudo_t[high_rel_mask],
                    domain="target",
                )

            l_align = self.align_loss_fn(
                ffusion_s,
                y_s,
                ffusion_t,
                pseudo_t,
                w_t,
                prototype_bank,
            )

            if use_sep:

                l_sep = self.sep_loss_fn(
                    ffusion_s,
                    y_s,
                    ffusion_t,
                    pseudo_t,
                    w_t,
                    prototype_bank,
                )

        # -----------------------------
        # DMSL
        # -----------------------------
        l_dmsl = torch.tensor(0.0, device=logits_s.device)

        if use_dmsl:

            l_dmsl = self.dmsl_loss_fn(
                logits_t,
                logits_t_teach,
                weights=w_t,
            )

        # -----------------------------
        # Total Loss
        # -----------------------------
        loss = (
            self.lambda_cls * l_cls
            + self.lambda_mgda * l_mgda
            + self.lambda_align * l_align
            + self.lambda_sep * l_sep
            + self.lambda_dmsl * l_dmsl
        )

        loss_dict = {
            "loss": loss.item(),
            "l_cls": l_cls.item(),
            "l_mgda": l_mgda.item(),
            "l_align": l_align.item(),
            "l_sep": l_sep.item(),
            "l_dmsl": l_dmsl.item(),
        }

        return loss, loss_dict
