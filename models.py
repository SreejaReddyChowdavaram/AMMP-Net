# -*- coding: utf-8 -*-
# models.py
# Implementation of AMMP-Net model modules, scans, prototype banks, and loss layers (Merged).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Directional Scan Helpers
# -------------------------------------------------------------------------
def get_diagonal_indices(H: int, W: int) -> list:
    """Generates coordinate sequence along grid diagonals."""
    indices = []
    for d in range(H + W - 1):
        for r in range(max(0, d - W + 1), min(H, d + 1)):
            c = d - r
            indices.append((r, c))
    return indices


def row_scan(x: torch.Tensor) -> torch.Tensor:
    """
    Flattens spatial representation row-by-row.
    Shape: (B, C, H, W) -> (B, H * W, C)
    """
    B, C, H, W = x.shape
    return x.permute(0, 2, 3, 1).reshape(B, H * W, C)


def row_reconstruct(seq: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Rebuilds spatial map from row-by-row sequence.
    Shape: (B, H * W, C) -> (B, C, H, W)
    """
    B, L, C = seq.shape
    return seq.reshape(B, H, W, C).permute(0, 3, 1, 2)


def column_scan(x: torch.Tensor) -> torch.Tensor:
    """
    Flattens spatial representation column-by-column.
    Shape: (B, C, H, W) -> (B, H * W, C)
    """
    B, C, H, W = x.shape
    return x.permute(0, 3, 2, 1).reshape(B, W * H, C)


def column_reconstruct(seq: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Rebuilds spatial map from column-by-column sequence.
    Shape: (B, H * W, C) -> (B, C, H, W)
    """
    B, L, C = seq.shape
    return seq.reshape(B, W, H, C).permute(0, 3, 2, 1)





def diagonal_reconstruct(seq: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Rebuilds spatial map from diagonal sequence.
    Shape: (B, H * W, C) -> (B, C, H, W)
    """
    B, L, C = seq.shape
    indices = get_diagonal_indices(H, W)
    out = torch.zeros(B, H, W, C, device=seq.device, dtype=seq.dtype)
    for idx, (r, c) in enumerate(indices):
        out[:, r, c, :] = seq[:, idx, :]
    return out.permute(0, 3, 1, 2)


def anti_diagonal_scan(x: torch.Tensor) -> torch.Tensor:
    """Flattens spatial representation along anti-diagonals."""
    B, C, H, W = x.shape
    x_flipped = torch.flip(x, dims=[3])
    indices = get_diagonal_indices(H, W)
    out = []
    for r, c in indices:
        out.append(x_flipped[:, :, r, c])
    return torch.stack(out, dim=1)


def anti_diagonal_reconstruct(seq: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Rebuilds spatial map from anti-diagonal sequence.
    Shape: (B, H * W, C) -> (B, C, H, W)
    """
    recon = diagonal_reconstruct(seq, H, W)
    return torch.flip(recon, dims=[3])


# -------------------------------------------------------------------------
# Shared Bidirectional Mamba (Pure PyTorch)
# -------------------------------------------------------------------------
class SharedBiMamba(nn.Module):
    """
    A pure PyTorch implementation of a Bidirectional Mamba / SSM block.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model)
        self.dt_proj = nn.Linear(d_model, d_model)
        
        self.A = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward_ssm(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state
        
        x_proj_out = self.x_proj(x)  # (B, L, 2 * N + D)
        B_param, C_param, dt = torch.split(x_proj_out, [N, N, D], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))  # (B, L, D)
        A = -torch.exp(self.A)             # (D, N)
        
        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            delta_t = dt[:, t, :].unsqueeze(-1)              # (B, D, 1)
            dA_t = torch.exp(delta_t * A.unsqueeze(0))       # (B, D, N)
            dB_t = delta_t * B_param[:, t, :].unsqueeze(1)   # (B, D, N)
            h = dA_t * h + dB_t * x[:, t, :].unsqueeze(-1)
            y_t = (h * C_param[:, t, :].unsqueeze(1)).sum(dim=-1)   # (B, D)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.in_proj(x)
        x_main, x_gate = torch.chunk(proj, 2, dim=-1)
        
        x_main = x_main.transpose(1, 2)
        x_main = F.silu(self.conv(x_main))
        x_main = x_main.transpose(1, 2)
        
        y_fwd = self.forward_ssm(x_main)
        
        x_bwd = torch.flip(x_main, dims=[1])
        y_bwd = torch.flip(self.forward_ssm(x_bwd), dims=[1])
        
        y = y_fwd + y_bwd
        out = y * F.silu(x_gate)
        return self.out_proj(out)


# -------------------------------------------------------------------------
# ADAM (Adaptive Direction-Aware Mamba)
# -------------------------------------------------------------------------
class ADAM(nn.Module):
    """
    Applies SharedBiMamba across 4 scanning directions and aggregates
    directional features dynamically using a spatial attention mechanism.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.shared_mamba = SharedBiMamba(d_model, d_state)
        # Global direction attention MLP: two linear layers (W1, W2)
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Scan in four directions
        s_row = row_scan(x)
        s_col = column_scan(x)
        s_diag = diagonal_scan(x)
        s_anti = anti_diagonal_scan(x)
        
        # Apply shared bidirectional Mamba to each sequence
        f_row = self.shared_mamba(s_row)   # (B, L, D)
        f_col = self.shared_mamba(s_col)
        f_diag = self.shared_mamba(s_diag)
        f_anti = self.shared_mamba(s_anti)
        
        # Reconstruct spatial feature maps
        h_row = row_reconstruct(f_row, H, W)   # (B, D, H, W)
        h_col = column_reconstruct(f_col, H, W)
        h_diag = diagonal_reconstruct(f_diag, H, W)
        h_anti = anti_diagonal_reconstruct(f_anti, H, W)
        
        # Global Average Pooling per direction to obtain descriptors g_k
        g_row = F.adaptive_avg_pool2d(h_row, 1).view(B, -1)  # (B, D)
        g_col = F.adaptive_avg_pool2d(h_col, 1).view(B, -1)
        g_diag = F.adaptive_avg_pool2d(h_diag, 1).view(B, -1)
        g_anti = F.adaptive_avg_pool2d(h_anti, 1).view(B, -1)
        
        # Compute attention logits e_k via shared MLP
        e_row = self.attn_mlp(g_row)  # (B, 1)
        e_col = self.attn_mlp(g_col)
        e_diag = self.attn_mlp(g_diag)
        e_anti = self.attn_mlp(g_anti)
        
        # Stack and softmax across directions
        e_stack = torch.cat([e_row, e_col, e_diag, e_anti], dim=1)  # (B,4)
        alpha = F.softmax(e_stack, dim=1)  # (B,4)
        
        # Fuse directional features weighted by alpha
        alpha = alpha.view(B, 4, 1, 1, 1)  # (B,4,1,1,1)
        stacked = torch.stack([h_row, h_col, h_diag, h_anti], dim=1)  # (B,4,D,H,W)
        fused = (stacked * alpha).sum(dim=1)  # (B,D,H,W)
        return fused


# -------------------------------------------------------------------------
# GC-DMF (Global-Context Directional Mamba Fusion)
# -------------------------------------------------------------------------
class GCDMF(nn.Module):
    """
    Fuses the original CNN features and the Direction-Aware Mamba features (ADAM)
    using a joint Mamba fusion block and a residual connection.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.fusion_mamba = SharedBiMamba(d_model, d_state)
        self.proj = nn.Conv2d(d_model * 2, d_model, kernel_size=1)
        
    def forward(self, spatial_feat: torch.Tensor, directional_feat: torch.Tensor) -> torch.Tensor:
        concat_feat = torch.cat([spatial_feat, directional_feat], dim=1)  # (B, 2*D, H, W)
        reduced_feat = self.proj(concat_feat)                            # (B, D, H, W)
        
        B, D, H, W = reduced_feat.shape
        seq = row_scan(reduced_feat)
        
        fused_seq = self.fusion_mamba(seq)
        
        fused_feat = row_reconstruct(fused_seq, H, W)
        return fused_feat + directional_feat


# -------------------------------------------------------------------------
# AMMP Backbone
# -------------------------------------------------------------------------
class AMMPBackbone(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, patch_size: int,
                 d_model: int = 64, d_state: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.d_model = d_model
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )
        
        self.adam = ADAM(d_model, d_state)
        self.gcdmf = GCDMF(d_model, d_state)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier_cnn = nn.Linear(d_model, num_classes)
        self.classifier_adam = nn.Linear(d_model, num_classes)
        self.classifier_fusion = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> tuple:
        f_cnn = self.cnn(x)                    # (B, D, H, W)
        f_adam = self.adam(f_cnn)              # (B, D, H, W)
        f_fusion = self.gcdmf(f_cnn, f_adam)   # (B, D, H, W)
        
        v_cnn = self.gap(f_cnn).squeeze(-1).squeeze(-1)       # (B, D)
        v_adam = self.gap(f_adam).squeeze(-1).squeeze(-1)     # (B, D)
        v_fusion = self.gap(f_fusion).squeeze(-1).squeeze(-1)  # (B, D)
        
        logits_cnn = self.classifier_cnn(v_cnn)
        logits_adam = self.classifier_adam(v_adam)
        logits_fusion = self.classifier_fusion(v_fusion)
        
        return logits_fusion, v_cnn, v_adam, v_fusion


# -------------------------------------------------------------------------
# Simple AMMP Model
# -------------------------------------------------------------------------
class SimpleAMMPModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, patch_size: int,
                 d_model: int = 64, d_state: int = 16):
        super().__init__()
        self.backbone = AMMPBackbone(in_channels, num_classes, patch_size, d_model, d_state)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _, _ = self.backbone(x)
        return logits




# -------------------------------------------------------------------------
# Momentum Prototype Bank
# -------------------------------------------------------------------------
class MomentumPrototypeBank(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int, momentum: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        self.register_buffer("src_prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("tgt_prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("src_initialized", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("tgt_initialized", torch.zeros(num_classes, dtype=torch.bool))
        
    def update(self, feats: torch.Tensor, labels: torch.Tensor, domain: str = "source"):
        prototypes = self.src_prototypes if domain == "source" else self.tgt_prototypes
        initialized = self.src_initialized if domain == "source" else self.tgt_initialized
        
        if feats.size(0) == 0:
            return
            
        feats_detached = feats.detach()
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                c_feats = feats_detached[mask]
                c_mean = c_feats.mean(dim=0)
                if initialized[c]:
                    prototypes[c] = self.momentum * prototypes[c] + (1.0 - self.momentum) * c_mean
                else:
                    prototypes[c] = c_mean
                    initialized[c] = True




# -------------------------------------------------------------------------
# Full AMMPNet
# -------------------------------------------------------------------------
class FullAMMPNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, patch_size: int,
                 d_model: int = 64, d_state: int = 16, memory_bank_momentum: float = 0.9):
        super().__init__()
        self.student = AMMPBackbone(in_channels, num_classes, patch_size, d_model, d_state)
        self.teacher = AMMPBackbone(in_channels, num_classes, patch_size, d_model, d_state)
        
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.prototype_bank = MomentumPrototypeBank(num_classes, d_model, momentum=memory_bank_momentum)
        
    def update_teacher(self, momentum: float = 0.99):
        for s_p, t_p in zip(self.student.parameters(), self.teacher.parameters()):
            t_p.data = momentum * t_p.data + (1.0 - momentum) * s_p.data
            
    def forward(self, x_s: torch.Tensor, y_s: torch.Tensor = None, x_t: torch.Tensor = None) -> tuple:
        if y_s is None and x_t is None:
            logits, _, _, _ = self.student(x_s)
            return logits
            
        logits_s, fcnn_s, fadam_s, ffusion_s = self.student(x_s)
        logits_t, fcnn_t, fadam_t, ffusion_t = self.student(x_t)
        
        with torch.no_grad():
            logits_t_teach, _, _, ffusion_t_teach = self.teacher(x_t)
            
        return (
            logits_s, fcnn_s, fadam_s, ffusion_s,
            logits_t, fcnn_t, fadam_t, ffusion_t,
            logits_t_teach, ffusion_t_teach
        )


