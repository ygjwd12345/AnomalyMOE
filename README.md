<div align="center">

# AnomalyMOE

**Mixture-of-Experts Bottleneck with Entropy-Based Routing for Unified Anomaly Detection**

Built on top of [Dinomaly](https://github.com/cnulab/Dinomaly) with DINOv3 encoder.

</div>

## Overview

AnomalyMOE replaces the single bottleneck MLP in the reconstruction-based anomaly detection pipeline with a **Mixture-of-Experts (MoE) Bottleneck**. A lightweight router dynamically assigns each input to specialized expert sub-networks, enabling expert-level specialization across diverse object categories in unified (multi-class) anomaly detection.

**Key contribution**: We propose an **Entropy-Based Routing Constraint (CE loss)** that replaces the traditional non-differentiable `f*P` load balancing loss. CE loss provides purely differentiable gradients to the router through:
- **Local Sharpness**: Minimizes per-sample routing entropy for confident expert assignment
- **Global Balance**: Maximizes batch-level entropy for uniform expert utilization  
- **Orthogonality**: Keeps router projection weights distinct

The router autonomously discovers **semantic visual clusters** (e.g., textures vs. round objects vs. small elongated objects), achieving expert-level specialization without any category labels during training.

## Architecture

```
Input Image
    |
[DINOv3 Encoder] (frozen)
    |
[Feature Fusion] (multi-layer aggregation)
    |
[MoE Bottleneck]
    |--- Router: context-based top-k expert selection
    |--- Expert 0: bMlp (specialized for category group A)
    |--- Expert 1: bMlp (specialized for category group B)
    |--- Expert 2: bMlp (specialized for category group C)
    |--- Expert 3: bMlp (specialized for category group D)
    |
[Decoder] (8-layer Linear Attention Transformer)
    |
[Anomaly Map] (encoder-decoder cosine distance)
```

## Results on MVTec AD (Unified)

| Method | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|--------|---------|------|------|---------|------|------|---------|
| Dinomaly-v3 (baseline) | 99.5 | 99.7 | 98.9 | 98.5 | 73.4 | 70.7 | 94.9 |
| **MoE-4E top3 + CE (50K)** | **99.6** | **99.8** | **99.0** | **98.6** | **74.6** | **71.3** | **95.2** |

## Quick Start

### Environment

```
CUDA >= 12.x
Python >= 3.10
PyTorch >= 2.0
```

### Installation

```bash
pip install -r requirements.txt
```

Download DINOv3 weights to `weights/`:
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

### Training

```bash
# Baseline (no MoE)
CUDA_VISIBLE_DEVICES=0 python dinomaly_mvtec_uni_base.py

# MoE Bottleneck with CE loss (recommended: 4 experts, top-3, 50K iters)
CUDA_VISIBLE_DEVICES=0 python dinomaly_mvtec_uni_base_moe.py \
    --save_name mvtec_E4T3_ce_50k

# VisA dataset
CUDA_VISIBLE_DEVICES=0 python dinomaly_visa_uni_moe.py \
    --save_name visa_E4T3_ce_50k
```

### Testing Saved Models

```bash
# Test baseline
python test_model.py --dataset mvtec \
    --model_path ./saved_results/mvtec_base/model.pth

# Test MoE model
python test_model.py --dataset mvtec \
    --model_path ./saved_results/mvtec_E4T3_ce_50k/model.pth \
    --use_moe --num_experts 4 --top_k 3
```

### Router Visualization

```bash
# All visualizations (needs trained MoE model + log files)
python visualize_router.py --mode all --dataset mvtec \
    --model_path ./saved_results/mvtec_E4T3_ce_50k/model.pth \
    --num_experts 4 --top_k 3 \
    --log_fp ./saved_results/mvtec_E4T3_fp/log.txt \
    --log_ce ./saved_results/mvtec_E4T3_ce_50k/log.txt

# Only heatmap + normal/anomaly analysis
python visualize_router.py --mode heatmap --dataset mvtec \
    --model_path ./saved_results/mvtec_E4T3_ce_50k/model.pth \
    --num_experts 4 --top_k 3

# Only routing evolution comparison
python visualize_router.py --mode evolution \
    --log_fp ./saved_results/mvtec_E4T3_fp/log.txt \
    --log_ce ./saved_results/mvtec_E4T3_ce_50k/log.txt

# Per-expert anomaly map decomposition
python visualize_router.py --mode expert_maps --dataset mvtec \
    --model_path ./saved_results/mvtec_E4T3_ce_50k/model.pth \
    --num_experts 4 --top_k 3 \
    --vis_categories carpet bottle transistor
```

Visualizations are saved to `./vis_results/` as both PNG images and Excel files for secondary plotting.

## Visualization Experiments

| Experiment | Description |
|-----------|-------------|
| **Exp 1**: Category-Expert Heatmap | 3-panel heatmap showing raw probability, top-k selection frequency, and contribution weight per category |
| **Exp 2**: Normal vs Anomaly | Compares routing distributions between normal and anomaly samples |
| **Exp 3**: Routing Evolution | Training curves comparing f\*P vs CE routing probability dynamics |
| **Exp 4**: Expert Anomaly Maps | Per-expert anomaly map decomposition for visual inspection |
| **Exp 5**: Router Weight Similarity | Cosine similarity matrix of router projection weights |

## MoE Configuration

Key hyperparameters in `dinomaly_mvtec_uni_base_moe.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 4 | Number of expert bMlp networks |
| `top_k` | 3 | Number of experts selected per sample |
| `routing_loss` | `'entropy'` | `'entropy'` (CE) or `'legacy'` (f\*P) |
| `lambda_local` | 0.5 | Local entropy sharpness weight |
| `lambda_global` | 0.5 | Global entropy balance weight |
| `lambda_ortho` | 0 | Router weight orthogonality weight |
| `total_iters` | 50000 | Training iterations |

## File Structure

```
.
├── models/
│   ├── uad.py                          # MoEBottleneck, ViTill, entropy routing
│   └── vision_transformer.py           # bMlp, Attention blocks
├── dinomaly_mvtec_uni_base.py          # MVTec baseline (no MoE)
├── dinomaly_mvtec_uni_base_moe.py      # MVTec with MoE + CE loss
├── dinomaly_visa_uni.py                # VisA baseline
├── dinomaly_visa_uni_moe.py            # VisA with MoE + CE loss
├── visualize_router.py                 # Router visualization toolkit
├── test_model.py                       # Evaluate saved checkpoints
├── dataset.py                          # MVTec/VisA data loading
└── utils.py                            # Evaluation, loss functions
```

## Acknowledgements

This project is built upon [Dinomaly](https://github.com/cnulab/Dinomaly).
