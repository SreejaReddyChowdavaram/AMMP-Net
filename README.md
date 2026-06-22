# ASDF-MambaNet++: Adaptive Spectral-Domain Fusion Mamba Network ++

> **State-of-the-Art Cross-Scene Hyperspectral Image Domain Adaptation**
> An advanced spatial-spectral dual-stream network integrating independent Multi-Directional Mamba encoders, Band-Attention multi-scale spectral feature learning, dynamic memory banks with Prototype Contrastive Loss, and Monte-Carlo Dropout uncertainty pseudo-label selection.

---

## 1. Key Innovations in ASDF-MambaNet++

1. **Independent Directional Mambas**: Row, Column, Diagonal, and Anti-Diagonal scans are modeled by independent state-space models to prevent scanning information collapse.
2. **Direction-Aware Attention Fusion**: Synthesizes the four directional representations using learned dynamic spatial weight scores.
3. **Spectral Mamba Plus Branch**: Preserves precise spectral signatures using 1D Multi-scale Spectral Convolutions, Band-wise Attention, and a Spectral CLS Token transformer-like sequence modeling.
4. **Cross-Attention Fusion Module**: Integrates local spatial, global spatial, spectral, and domain-invariant features using multi-head query-key attention.
5. **No Global MMD**: Avoids negative transfer by replacing global MMD with Class-Conditional MMD and Prototype Guided Alignment.
6. **Dynamic Memory Bank**: Maintains historical source and target prototypes using class-wise FIFO queues updated via momentum.
7. **Prototype Contrastive Loss**: Minimizes cross-domain prototype distance while maximizing inter-class separation.
8. **Uncertainty Guided Pseudo-Labeling**: Filters target predictions using Monte-Carlo Dropout prediction variance, Shannon entropy thresholds, and temporal ensemble consistency.
9. **Class Imbalance Solutions**: Integrates a Class Balanced Sampler and Focal Loss with Adaptive Class Weighting.
10. **4-Stage Adaptive Curriculum Learning**: Transitions from source-only pretraining, to weak class alignment, prototype refinement, and self-distillation.

---

## 2. Directory Structure

```
UM2PMA_Net/
├── train_new.py           ← Main training script with 4-stage scheduling
├── models.py              ← ASDF-MambaNet++ architecture definition
├── dataset.py             ← Class-balanced sampling, patch loader
├── losses.py              ← Prototype contrastive, CC-MMD, memory, consistency losses
├── utils.py               ← Plotting helpers, sklearn metrics evaluation
├── evaluation.py          ← Checkpoint evaluation
├── dataset_registry.py    ← Dynamic dataset registry
├── config.py              ← Hyperparameter configurations
├── README.md              ← Setup & run instructions
├── ARCHITECTURE.md        ← Architectural design and tensor shapes
├── WORKFLOW.md            ← 4-stage dataflow and pipeline
├── ABLATION_STUDY.md      ← Ablation study design and results
└── PAPER_NOTES.md         ← Journal submission and writing notes
```

---

## 3. Installation & Setup

```bash
# Create conda env
conda create -n asdfplus python=3.9 -y
conda activate asdfplus

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Training Commands

Every run automatically creates a unique directory structure under `results/{Dataset}/{Timestamp}/` saving all curves, log files, reports, and models.

```bash
# Run training on Houston13 -> Houston18
python train_new.py --dataset houston --epochs 50

# Run training on HyRANK Dioni -> Loukia
python train_new.py --dataset hyrank --epochs 50

# Custom hyperparameter overrides
python train_new.py --dataset houston --epochs 50 --lr 3e-4 --batch_size 32
```

---

## 5. Evaluation

```bash
python evaluation.py --checkpoint results/Houston/20260721_101530/best_model.pth --dataset houston
```

---

## 6. Performance Benchmarks

| Dataset | Metric | BMDMNet | **ASDF-MambaNet++ (Target)** |
|---|---|---|---|
| **Houston13 → Houston18** | OA (%) | 77.93 | **> 80.00** |
| | AA (%) | 56.64 | **> 70.00** |
| | Kappa (%) | 62.76 | **> 68.00** |
| **HyRANK Dioni → Loukia** | OA (%) | 65.06 | **> 68.00** |
| | AA (%) | 53.82 | **> 58.00** |
| | Kappa (%) | 57.03 | **> 60.00** |
