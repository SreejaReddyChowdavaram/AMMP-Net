# -*- coding: utf-8 -*-
# dataset.py
# Dataset loading, patch extraction, HSI dataset class, and DataLoader construction for AMMP-Net.

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# -------------------------------------------------------------------------
# Normalization and shape alignment
# -------------------------------------------------------------------------
def normalize_hsi(data: np.ndarray) -> np.ndarray:
    """
    Min-max normalize HSI data to range [0, 1] band-by-band or globally.
    """
    data = data.astype(np.float32)
    lo, hi = data.min(), data.max()
    if hi - lo < 1e-8:
        return np.zeros_like(data)
    return (data - lo) / (hi - lo + 1e-8)


def align_data_to_hwc(data: np.ndarray, expected_c: int) -> np.ndarray:
    """
    Ensures the channels dimension is at the last axis (H, W, C).
    """
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data.shape}")
    s = data.shape
    if s[2] == expected_c:
        return data
    for perm in [(1, 2, 0), (0, 2, 1)]:
        cand = np.transpose(data, perm)
        if cand.shape[2] == expected_c:
            return cand
    c_idx = int(np.argmin(s))
    others = [i for i in range(3) if i != c_idx]
    return np.transpose(data, others + [c_idx])


def align_gt_to_hw(gt: np.ndarray, spatial: tuple) -> np.ndarray:
    """
    Aligns ground truth label matrix to the HSI spatial dimensions (H, W).
    """
    gt = np.squeeze(gt).astype(np.int32)
    if gt.ndim == 1:
        H, W = spatial
        if gt.size == H * W:
            return gt.reshape(H, W)
    if gt.ndim == 2:
        if gt.shape == spatial:
            return gt
        if gt.shape[::-1] == spatial:
            return gt.T
        return gt
    if gt.ndim == 3:
        gt = gt[0] if gt.shape[0] < gt.shape[2] else gt[:, :, 0]
        return align_gt_to_hw(gt, spatial)
    raise ValueError(f"Cannot align GT with shape {gt.shape} to spatial {spatial}")


# -------------------------------------------------------------------------
# Patch extraction (Vectorized)
# -------------------------------------------------------------------------
def extract_patches(data: np.ndarray, gt: np.ndarray, patch_size: int):
    """
    Extracts patch_size x patch_size patches around all labeled HSI pixels.
    Uses vectorized unfold padding for high performance.
    """
    assert data.ndim == 3 and gt.ndim == 2
    assert data.shape[:2] == gt.shape, \
        f"Shape mismatch between HSI {data.shape[:2]} and GT {gt.shape}"
    H, W, C = data.shape
    m = patch_size // 2
    
    # Pad HSI data with reflection padding
    data_t = torch.from_numpy(data).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
    padded = torch.nn.functional.pad(data_t, (m, m, m, m), mode="reflect")
    
    # Locate labeled coordinates (labels > 0)
    gt_t = torch.from_numpy(gt)
    rows, cols = torch.where(gt_t > 0)
    N = len(rows)
    if N == 0:
        raise ValueError("Ground truth has no labeled pixels")
        
    # Vectorized unfold extract
    patches_unfolded = padded.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    # patches_unfolded: (1, C, H_out, W_out, P, P)
    patches = patches_unfolded[0, :, rows, cols, :, :].permute(1, 2, 3, 0) # (N, P, P, C)
    labels = gt_t[rows, cols] - 1  # Map to 0-indexed classes
    
    return patches.numpy(), labels.numpy().astype(np.int32), torch.stack([rows, cols], dim=1).numpy()


# -------------------------------------------------------------------------
# Stratified sampling
# -------------------------------------------------------------------------
def stratified_sample(patches, labels, train_ratio=0.05, min_per_class=5, seed=42):
    """
    Performs stratified class-balanced split.
    """
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        cidx = np.where(labels == c)[0]
        n = min(max(min_per_class, int(len(cidx) * train_ratio)), len(cidx))
        idx.extend(rng.choice(cidx, n, replace=False).tolist())
    idx = np.array(idx, dtype=np.int64)
    return patches[idx], labels[idx]


# -------------------------------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------------------------------
class HSIDataset(Dataset):
    """
    Dataset representing hyperspectral image patches.
    Supports random flips, rotations, and channel-wise noise for augmentation.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray = None,
                 is_train: bool = True, aug_prob: float = 0.5, noise_std: float = 0.01,
                 return_idx: bool = False):
        # Convert HWC to CHW: (N, P, P, C) -> (N, C, P, P)
        self.X = np.ascontiguousarray(X.transpose(0, 3, 1, 2)).astype(np.float32)
        self.Y = Y
        self.is_train = is_train
        self.aug_prob = aug_prob
        self.noise_std = noise_std
        self.return_idx = return_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        
        # Apply data augmentations for training
        if self.is_train:
            if np.random.rand() < self.aug_prob:
                x = np.flip(x, axis=2).copy()  # Horizontal flip
            if np.random.rand() < self.aug_prob:
                x = np.flip(x, axis=1).copy()  # Vertical flip
            if np.random.rand() < self.aug_prob:
                x = np.rot90(x, k=np.random.randint(1, 4), axes=(1, 2)).copy()  # Rotations
            
            # Amplitude scaling and Gaussian noise injection
            alpha = np.random.uniform(0.95, 1.05)
            noise = np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
            x = np.clip(alpha * x + noise, 0.0, 1.5)
            
        x_t = torch.from_numpy(x)
        
        if self.return_idx:
            if self.Y is not None:
                return x_t, torch.tensor(int(self.Y[idx]), dtype=torch.long), torch.tensor(idx, dtype=torch.long)
            return x_t, torch.tensor(idx, dtype=torch.long)
        else:
            if self.Y is not None:
                return x_t, torch.tensor(int(self.Y[idx]), dtype=torch.long)
            return x_t


# -------------------------------------------------------------------------
# Synthetic Dataset Generator for Fallbacks
# -------------------------------------------------------------------------
def generate_synthetic_hsi(in_channels: int, num_classes: int,
                           height: int = 120, width: int = 120, seed: int = 42):
    """
    Generates a synthetic hyperspectral image with structured class regions.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.1, (height, width, in_channels)).astype(np.float32)
    gt = np.zeros((height, width), dtype=np.int32)
    
    # Define prototypes for each class
    protos = rng.uniform(-1.0, 1.0, (num_classes + 1, in_channels))
    
    # Partition space into class blocks
    rows = max(2, int(np.ceil(np.sqrt(num_classes))))
    cols = max(2, int(np.ceil(num_classes / rows)))
    gh, gw = height // rows, width // cols
    
    class_idx = 1
    for r in range(rows):
        for col in range(cols):
            if class_idx > num_classes:
                break
            h0, h1 = r * gh, min((r + 1) * gh, height)
            w0, w1 = col * gw, min((col + 1) * gw, width)
            gt[h0:h1, w0:w1] = class_idx
            data[h0:h1, w0:w1, :] += protos[class_idx]
            class_idx += 1
            
    # Apply a spatial Gaussian filter to simulate spatial correlation
    try:
        from scipy.ndimage import gaussian_filter
        for b in range(in_channels):
            data[:, :, b] = gaussian_filter(data[:, :, b], sigma=1.5)
    except ImportError:
        pass
        
    return normalize_hsi(data), gt


# -------------------------------------------------------------------------
# Load individual MAT files
# -------------------------------------------------------------------------
def load_mat_file(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return sio.loadmat(path)
    except Exception as e:
        # Fallback to h5py for v7.3 MAT files
        try:
            import h5py
            f = h5py.File(path, "r")
            return {k: np.array(v) for k, v in f.items() if not k.startswith("_")}
        except ImportError:
            raise IOError(f"Cannot read {path}. Verify it is a valid MAT file and h5py is installed if v7.3.") from e


def extract_largest_array(mat: dict) -> np.ndarray:
    """Extracts the largest data array from MAT dictionary (ignores metadata keys)."""
    keys = [k for k in mat.keys() if not k.startswith("_")]
    if not keys:
        raise ValueError("No valid variables found in MAT file")
    return np.array(mat[max(keys, key=lambda k: np.array(mat[k]).size)])


def load_domain_data(mat_path: str, gt_path: str, in_channels: int, name: str):
    """
    Loads and aligns the HSI cube and corresponding GT labels for a domain.
    """
    data = gt = None
    try:
        d = load_mat_file(mat_path)
        data = align_data_to_hwc(extract_largest_array(d), in_channels)
        print(f"[{name}] HSI shape: {data.shape} loaded from {os.path.basename(mat_path)}")
    except Exception as e:
        print(f"[{name}] HSI load failed: {e}")
        
    if data is not None:
        try:
            g = load_mat_file(gt_path)
            gt = align_gt_to_hw(extract_largest_array(g), data.shape[:2])
            unique_lbls = np.unique(gt[gt > 0])
            print(f"[{name}] GT shape: {gt.shape}, unique classes: {len(unique_lbls)} loaded from {os.path.basename(gt_path)}")
        except Exception as e:
            print(f"[{name}] GT load failed: {e}")
            data = None
            
    return data, gt


# -------------------------------------------------------------------------
# Main Loader Interface
# -------------------------------------------------------------------------
def get_data_loaders(cfg: dict, train_ratio: float = 0.05, seed: int = 42,
                     max_target_samples: int = None, cache_dataset: bool = True):
    """
    Builds the source train, target train, and target test/validation DataLoaders.
    Includes automated dataset caching, class balancing, and synthetic data fallbacks.
    """
    name = cfg["dataset_name"]
    in_ch = cfg["in_channels"]
    n_cls = cfg["num_classes"]
    P = cfg["patch_size"]
    bs = cfg["batch_size"]

    print("=" * 60)
    print(f"  PREPARING CROSS-SCENE DATASET: {name}")
    print("=" * 60)

    # ── Dataset Caching ──────────────────────────────────────────────────────
    npz_cache_path = os.path.join(".", f"{name.lower()}_patches.npz")
    cache_loaded = False
    Xs = Ys = Xt = Yt = None

    if cache_dataset and os.path.exists(npz_cache_path):
        print(f"[Dataset Cache] Loading pre-extracted patches from {npz_cache_path}...")
        try:
            data_cache = np.load(npz_cache_path)
            Xs = data_cache["Xs"]
            Ys = data_cache["Ys"]
            Xt = data_cache["Xt"]
            Yt = data_cache["Yt"]
            print(f"[Dataset Cache] Load successful. Source patches: {Xs.shape[0]}, Target patches: {Xt.shape[0]}")
            cache_loaded = True
        except Exception as e:
            print(f"[Dataset Cache] Load failed: {e}. Re-extracting from raw files...")
            Xs = Ys = Xt = Yt = None

    if not cache_loaded:
        # Load raw source domain
        src_data, src_gt = load_domain_data(cfg["source_path"], cfg["source_gt"], in_ch, f"{name}/Source")
        if src_data is None:
            print(f"[Dataset Warning] Raw source data load failed. Generating synthetic fallback...")
            src_data, src_gt = generate_synthetic_hsi(in_ch, n_cls, seed=seed)

        # Load raw target domain
        same_paths = (cfg["source_path"] == cfg["target_path"])
        if same_paths and not cfg.get("spatial_split", False):
            tgt_data, tgt_gt = src_data.copy(), src_gt.copy()
        else:
            tgt_data, tgt_gt = load_domain_data(cfg["target_path"], cfg["target_gt"], in_ch, f"{name}/Target")
            if tgt_data is None:
                print(f"[Dataset Warning] Raw target data load failed. Generating synthetic fallback...")
                tgt_data, tgt_gt = generate_synthetic_hsi(in_ch, n_cls, seed=seed + 1)
                
        # Handle spatial split option (split single scene into left-source and right-target)
        if cfg.get("spatial_split", False):
            print(f"[Dataset] Applying spatial split with ratio {cfg.get('split_ratio', 0.5)}")
            split_idx = int(src_gt.shape[1] * cfg.get("split_ratio", 0.5))
            src_gt = src_gt.copy()
            tgt_gt = tgt_gt.copy()
            src_gt[:, split_idx:] = 0
            tgt_gt[:, :split_idx] = 0

        # Channel check and update
        actual_ch = src_data.shape[2]
        if actual_ch != in_ch:
            print(f"[Dataset] Channel count discrepancy: config expected {in_ch}, found {actual_ch}. Updating config...")
            in_ch = actual_ch
            cfg["in_channels"] = in_ch

        # Perform HSI min-max scaling
        src_data = normalize_hsi(src_data)
        tgt_data = normalize_hsi(tgt_data)

        # Extract patch blocks
        Xs, Ys, _ = extract_patches(src_data, src_gt, P)
        Xt, Yt, _ = extract_patches(tgt_data, tgt_gt, P)
        
        # Save patches to cache
        if cache_dataset:
            try:
                np.savez_compressed(npz_cache_path, Xs=Xs, Ys=Ys, Xt=Xt, Yt=Yt)
                print(f"[Dataset Cache] Saved patches to {npz_cache_path}")
            except Exception as e:
                print(f"[Dataset Cache] Save failed: {e}")

    # Dynamic Class Count Alignment
    actual_cls = n_cls
    if len(Ys) > 0 and len(Yt) > 0:
        actual_cls = int(max(Ys.max(), Yt.max())) + 1
    if actual_cls != n_cls:
        print(f"[Dataset] Updating label count from {n_cls} to {actual_cls}")
        cfg["num_classes"] = actual_cls

    # Stratified Sampling on Source
    Xs_tr, Ys_tr = stratified_sample(Xs, Ys, train_ratio=train_ratio, seed=seed)
    
    # Ensure source dataset is large enough to avoid frequent loader iteration restarts
    min_samples = bs * 10
    if Xs_tr.shape[0] < min_samples:
        reps = int(np.ceil(min_samples / Xs_tr.shape[0]))
        Xs_tr = np.tile(Xs_tr, (reps, 1, 1, 1))
        Ys_tr = np.tile(Ys_tr, reps)

    # Target Subsampling for faster domain training steps if requested
    target_limit = max_target_samples if max_target_samples is not None else cfg.get("max_target_samples", 5000)
    if target_limit is not None and Xt.shape[0] > target_limit:
        print(f"[Dataset] Target Subsampling enabled: limiting training samples to {target_limit} (from {Xt.shape[0]})")
        rng = np.random.default_rng(seed + 99)
        sel_idx = rng.choice(Xt.shape[0], target_limit, replace=False)
        Xt_train = Xt[sel_idx]
    else:
        Xt_train = Xt

    # Construct weighted class sampler for source dataset to resolve imbalance
    unique_classes, class_counts = np.unique(Ys_tr, return_counts=True)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = np.array([class_weights[np.where(unique_classes == y)[0][0]] for y in Ys_tr])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Construct PyTorch Datasets
    src_ds       = HSIDataset(Xs_tr, Ys_tr, is_train=True)
    tgt_train_ds = HSIDataset(Xt_train, None,   is_train=True, return_idx=True)
    tgt_test_ds  = HSIDataset(Xt,    Yt,     is_train=False, return_idx=False)

    # Set appropriate worker options
    num_workers = 0  # 0 is safest and avoids multiprocess deadlock on Windows
    pin_memory = torch.cuda.is_available()
    
    test_bs = cfg.get("test_batch_size", 256)

    src_loader       = DataLoader(src_ds,       batch_size=bs, sampler=sampler,
                                  drop_last=True,  num_workers=num_workers, pin_memory=pin_memory)
    tgt_train_loader = DataLoader(tgt_train_ds, batch_size=bs, shuffle=True,
                                  drop_last=True,  num_workers=num_workers, pin_memory=pin_memory)
    tgt_test_loader  = DataLoader(tgt_test_ds,  batch_size=test_bs, shuffle=False,
                                  drop_last=False, num_workers=num_workers, pin_memory=pin_memory)

    dataset_info = {
        "dataset_name": name,
        "in_channels": in_ch,
        "num_classes": cfg["num_classes"],
        "patch_size": P,
        "num_source_train": Xs_tr.shape[0],
        "num_target_test": Xt.shape[0],
    }

    print("-" * 60)
    print(f"  Dataset Name       : {name}")
    print(f"  Bands / Channels   : {in_ch}")
    print(f"  Number of Classes  : {cfg['num_classes']}")
    print(f"  Source Train Size  : {Xs_tr.shape[0]}")
    print(f"  Target Test Size   : {Xt.shape[0]}")
    print(f"  Output batch size  : {bs}")
    print("=" * 60)

    return src_loader, tgt_train_loader, tgt_test_loader, Ys_tr, dataset_info, cfg
