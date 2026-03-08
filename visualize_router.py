"""
Router 有效性可视化实验工具。
生成 4 类图表来证明 MoE Router 是否真正在学习语义路由。

实验 1: Category-Expert 路由偏好热力图
实验 2: Normal vs Anomaly 路由差异分析
实验 3: 训练过程路由概率演化曲线 (从 log 文件解析)
实验 4: 单 Expert Anomaly Map 分解对比

用法:
  # 实验 1+2: Category-Expert 热力图 + Normal vs Anomaly (需要训练好的模型)
  python visualize_router.py --mode heatmap --dataset mvtec \
      --model_path ./saved_results/mvtec_E4T3_ce/model.pth \
      --num_experts 4 --top_k 3

  # 实验 3: 路由演化曲线 (需要 log 文件)
  python visualize_router.py --mode evolution \
      --log_fp ./saved_results/mvtec_E4T3_fp/log.txt \
      --log_ce ./saved_results/mvtec_E4T3_ce/log.txt

  # 实验 4: 单 Expert Anomaly Map 分解 (需要训练好的模型)
  python visualize_router.py --mode expert_maps --dataset mvtec \
      --model_path ./saved_results/mvtec_E4T3_ce/model.pth \
      --num_experts 4 --top_k 3 --vis_categories bottle carpet transistor

  # 全部一起跑
  python visualize_router.py --mode all --dataset mvtec \
      --model_path ./saved_results/mvtec_E4T3_ce/model.pth \
      --num_experts 4 --top_k 3 \
      --log_fp ./saved_results/mvtec_E4T3_fp/log.txt \
      --log_ce ./saved_results/mvtec_E4T3_ce/log.txt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import argparse
import warnings
import math
from functools import partial

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models.uad import ViTill, MoEBottleneck
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dataset import get_data_transforms, MVTecDataset
from utils import cal_anomaly_maps, get_gaussian_kernel
from dinov3.hub.backbones import load_dinov3_model

warnings.filterwarnings("ignore")

MVTEC_ITEMS = ['carpet', 'grid', 'leather', 'tile', 'wood',
               'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
               'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

VISA_ITEMS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
              'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

TEXTURE_CATEGORIES = {'carpet', 'grid', 'leather', 'tile', 'wood'}
OBJECT_CATEGORIES = {'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                     'pill', 'screw', 'toothbrush', 'transistor', 'zipper'}


def build_model(args, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = load_dinov3_model(
        'dinov3_vitb16', layers_to_extract_from=target_layers,
        pretrained_weight_path='weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')

    embed_dim, num_heads = 768, 12

    bottleneck = MoEBottleneck(
        dim=embed_dim, hidden_dim=embed_dim * 4,
        num_experts=args.num_experts, top_k=args.top_k,
        prompt_k=16, dropout=0.2,
        routing_loss=args.routing_loss,
        lambda_local=0.5, lambda_global=0.5, lambda_ortho=0.001,
    )

    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=LinearAttention2)
        for _ in range(8)
    ])

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=target_layers, mask_neighbor_size=0,
                   fuse_layer_encoder=fuse_layer_encoder,
                   fuse_layer_decoder=fuse_layer_decoder)

    state_dict = torch.load(args.model_path, map_location='cpu')
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Model loaded from {args.model_path}')
    if msg.missing_keys:
        print(f'  Missing keys: {msg.missing_keys}')
    if msg.unexpected_keys:
        print(f'  Unexpected keys: {msg.unexpected_keys}')

    model = model.to(device)
    model.eval()
    return model


# ======================================================================
# Experiment 1: Category-Expert Routing Preference Heatmap
# ======================================================================
def collect_routing_stats(model, data_path, item_list, device, batch_size=16, image_size=512, crop_size=448):
    """Collect per-sample routing probabilities AND top-k indices for every category."""
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    num_experts = model.bottleneck.num_experts
    top_k = model.bottleneck.top_k

    stats = {}
    for item in item_list:
        test_path = os.path.join(data_path, item)
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
        loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

        all_probs = []
        all_topk = []
        all_topk_weights = []
        all_labels = []
        with torch.no_grad():
            for img, gt, label, _ in loader:
                img = img.to(device)
                output = model(img)
                router_info = output[2]
                all_probs.append(router_info['router_probs'].cpu())
                all_topk.append(router_info['topk_indices'].cpu())
                all_topk_weights.append(router_info['topk_weights'].cpu())
                all_labels.append(label)

        all_probs = torch.cat(all_probs, dim=0).numpy()          # [N, num_experts]
        all_topk = torch.cat(all_topk, dim=0).numpy()            # [N, top_k]
        all_topk_weights = torch.cat(all_topk_weights, dim=0).numpy()  # [N, num_experts] renormalized
        all_labels = torch.cat(all_labels, dim=0).numpy()        # [N]

        # Top-K selection frequency: how often each expert appears in top-k
        topk_freq = np.zeros(num_experts)
        for e in range(num_experts):
            topk_freq[e] = np.mean(np.any(all_topk == e, axis=1))

        # Per normal/anomaly split
        normal_mask = all_labels == 0
        anomaly_mask = all_labels == 1

        stats[item] = {
            'probs': all_probs,
            'topk_indices': all_topk,
            'topk_freq': topk_freq,
            'topk_weights_mean': all_topk_weights.mean(axis=0),
            'labels': all_labels,
            'mean_probs': all_probs.mean(axis=0),
            'normal_probs': all_probs[normal_mask].mean(axis=0) if normal_mask.any() else None,
            'anomaly_probs': all_probs[anomaly_mask].mean(axis=0) if anomaly_mask.any() else None,
            'normal_topk_freq': None,
            'anomaly_topk_freq': None,
        }

        if normal_mask.any():
            nf = np.zeros(num_experts)
            for e in range(num_experts):
                nf[e] = np.mean(np.any(all_topk[normal_mask] == e, axis=1))
            stats[item]['normal_topk_freq'] = nf

        if anomaly_mask.any():
            af = np.zeros(num_experts)
            for e in range(num_experts):
                af[e] = np.mean(np.any(all_topk[anomaly_mask] == e, axis=1))
            stats[item]['anomaly_topk_freq'] = af

        print(f'  {item}: {len(all_probs)} samples, '
              f'P_mean=[{", ".join(f"{p:.3f}" for p in stats[item]["mean_probs"])}], '
              f'topk_freq=[{", ".join(f"{f:.2f}" for f in topk_freq)}]')

    return stats


def _draw_heatmap(ax, mat, item_list, title, cmap='YlOrRd', vmin=0, vmax=None, fmt='.3f'):
    """Helper: draw a single heatmap panel."""
    num_experts = mat.shape[1]
    if vmax is None:
        vmax = mat.max()
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f'E{i}' for i in range(num_experts)], fontsize=9)
    ax.set_yticks(range(len(item_list)))

    ylabels = []
    for item in item_list:
        tag = '[T]' if item in TEXTURE_CATEGORIES else '[O]'
        ylabels.append(f'{tag} {item}')
    ax.set_yticklabels(ylabels, fontsize=8)

    for i in range(len(item_list)):
        for j in range(num_experts):
            val = mat[i, j]
            color = 'white' if val > (vmin + (vmax - vmin) * 0.6) else 'black'
            ax.text(j, i, f'{val:{fmt}}', ha='center', va='center', fontsize=7, color=color)

    ax.set_title(title, fontsize=10, pad=8)
    return im


def plot_category_expert_heatmap(stats, item_list, save_path, top_k=3, title_suffix=''):
    """
    Experiment 1: Three-panel heatmap.
      (a) Raw Softmax Probability  - shows router's confidence distribution
      (b) Top-K Selection Frequency - how often each expert is selected (0~1)
      (c) Top-K Contribution Weight - renormalized weights among selected experts
    """
    num_experts = len(list(stats.values())[0]['mean_probs'])
    mat_prob = np.array([stats[item]['mean_probs'] for item in item_list])
    mat_freq = np.array([stats[item]['topk_freq'] for item in item_list])
    mat_weight = np.array([stats[item]['topk_weights_mean'] for item in item_list])

    fig, axes = plt.subplots(1, 3, figsize=(4.5 * 3, 1 + len(item_list) * 0.42))

    im0 = _draw_heatmap(axes[0], mat_prob, item_list,
                         '(a) Raw Router Probability', cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im0, ax=axes[0], shrink=0.6)

    im1 = _draw_heatmap(axes[1], mat_freq, item_list,
                         f'(b) Top-{top_k} Selection Frequency', cmap='Blues', vmin=0, vmax=1, fmt='.2f')
    plt.colorbar(im1, ax=axes[1], shrink=0.6)

    im2 = _draw_heatmap(axes[2], mat_weight, item_list,
                         f'(c) Top-{top_k} Contribution Weight', cmap='YlOrRd', vmin=0, fmt='.3f')
    plt.colorbar(im2, ax=axes[2], shrink=0.6)

    fig.suptitle(f'Category-Expert Routing Analysis (top_k={top_k}){title_suffix}', fontsize=13, y=1.02)
    plt.tight_layout()
    fpath = os.path.join(save_path, 'exp1_category_expert_heatmap.png')
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fpath}')

    print(f'  [Panel a] Raw prob row-std={mat_prob.std(axis=1).mean():.4f} '
          f'(higher = sharper per-category routing)')
    print(f'  [Panel b] Selection freq row-std={mat_freq.std(axis=1).mean():.4f} '
          f'(shows which experts actually participate)')
    print(f'  [Panel c] Contribution weight row-std={mat_weight.std(axis=1).mean():.4f} '
          f'(effective expert weighting)')

    # Summarize expert clusters
    primary_expert = mat_prob.argmax(axis=1)
    print(f'\n  Expert assignment summary:')
    for e in range(num_experts):
        assigned = [item_list[i] for i in range(len(item_list)) if primary_expert[i] == e]
        if assigned:
            types = [('[T]' if c in TEXTURE_CATEGORIES else '[O]') for c in assigned]
            print(f'    Expert {e}: {", ".join(f"{t}{c}" for t, c in zip(types, assigned))}')

    # Save to Excel
    ecols = [f'E{i}' for i in range(num_experts)]
    ylabels = [('[T] ' if item in TEXTURE_CATEGORIES else '[O] ') + item for item in item_list]

    with pd.ExcelWriter(os.path.join(save_path, 'exp1_heatmap_data.xlsx'), engine='openpyxl') as writer:
        pd.DataFrame(mat_prob, index=ylabels, columns=ecols).to_excel(writer, sheet_name='raw_probability')
        pd.DataFrame(mat_freq, index=ylabels, columns=ecols).to_excel(writer, sheet_name='selection_frequency')
        pd.DataFrame(mat_weight, index=ylabels, columns=ecols).to_excel(writer, sheet_name='contribution_weight')

        summary = pd.DataFrame({
            'category': item_list,
            'type': ['texture' if c in TEXTURE_CATEGORIES else 'object' for c in item_list],
            'primary_expert': primary_expert,
        })
        summary.to_excel(writer, sheet_name='expert_assignment', index=False)

    print(f'  Data saved: {os.path.join(save_path, "exp1_heatmap_data.xlsx")}')


# ======================================================================
# Experiment 2: Normal vs Anomaly Routing Difference
# ======================================================================
def plot_normal_vs_anomaly(stats, item_list, save_path):
    """Experiment 2: Compare routing distributions of normal vs anomaly samples.
    Shows two rows: raw probability and top-k selection frequency."""
    num_experts = len(list(stats.values())[0]['mean_probs'])

    items_with_both = [item for item in item_list
                       if stats[item]['normal_probs'] is not None
                       and stats[item]['anomaly_probs'] is not None]

    if len(items_with_both) == 0:
        print('  No categories have both normal and anomaly samples. Skipping.')
        return

    ncols = min(5, len(items_with_both))
    nrows_cat = (len(items_with_both) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows_cat, ncols, figsize=(ncols * 3.6, nrows_cat * 3))
    if nrows_cat == 1:
        axes = axes[np.newaxis, :] if ncols > 1 else np.array([[axes]])
    if ncols == 1:
        axes = axes[:, np.newaxis]

    global_normal_p = []
    global_anomaly_p = []
    global_normal_f = []
    global_anomaly_f = []

    for idx, item in enumerate(items_with_both):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        normal_p = stats[item]['normal_probs']
        anomaly_p = stats[item]['anomaly_probs']
        normal_f = stats[item]['normal_topk_freq']
        anomaly_f = stats[item]['anomaly_topk_freq']

        global_normal_p.append(normal_p)
        global_anomaly_p.append(anomaly_p)
        if normal_f is not None:
            global_normal_f.append(normal_f)
        if anomaly_f is not None:
            global_anomaly_f.append(anomaly_f)

        x = np.arange(num_experts)
        w = 0.18

        ax.bar(x - 1.5*w, normal_p, w, label='Normal (prob)', color='#2196F3', alpha=0.9)
        ax.bar(x - 0.5*w, anomaly_p, w, label='Anomaly (prob)', color='#F44336', alpha=0.9)
        if normal_f is not None:
            ax.bar(x + 0.5*w, normal_f, w, label='Normal (freq)', color='#90CAF9', alpha=0.7)
        if anomaly_f is not None:
            ax.bar(x + 1.5*w, anomaly_f, w, label='Anomaly (freq)', color='#EF9A9A', alpha=0.7)

        ax.set_title(item, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f'E{i}' for i in range(num_experts)], fontsize=8)
        ax.set_ylim(0, 1.15)
        if idx == 0:
            ax.legend(fontsize=5.5, loc='upper right')

    for idx in range(len(items_with_both), nrows_cat * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle('Normal vs Anomaly: Routing Probability & Top-K Selection Frequency',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fpath = os.path.join(save_path, 'exp2_normal_vs_anomaly.png')
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fpath}')

    if global_normal_p:
        gn_p = np.array(global_normal_p).mean(axis=0)
        ga_p = np.array(global_anomaly_p).mean(axis=0)
        diff_p = np.abs(gn_p - ga_p).sum()
        print(f'  Prob  - Normal: [{", ".join(f"{p:.3f}" for p in gn_p)}] | '
              f'Anomaly: [{", ".join(f"{p:.3f}" for p in ga_p)}] | L1={diff_p:.4f}')
    if global_normal_f:
        gn_f = np.array(global_normal_f).mean(axis=0)
        ga_f = np.array(global_anomaly_f).mean(axis=0)
        diff_f = np.abs(gn_f - ga_f).sum()
        print(f'  Freq  - Normal: [{", ".join(f"{p:.3f}" for p in gn_f)}] | '
              f'Anomaly: [{", ".join(f"{p:.3f}" for p in ga_f)}] | L1={diff_f:.4f}')

    # Save to Excel
    ecols = [f'E{i}' for i in range(num_experts)]
    rows_data = []
    for item in items_with_both:
        row = {'category': item, 'type': 'texture' if item in TEXTURE_CATEGORIES else 'object'}
        np_ = stats[item]['normal_probs']
        ap_ = stats[item]['anomaly_probs']
        nf_ = stats[item]['normal_topk_freq']
        af_ = stats[item]['anomaly_topk_freq']
        for i in range(num_experts):
            row[f'normal_prob_E{i}'] = np_[i] if np_ is not None else None
            row[f'anomaly_prob_E{i}'] = ap_[i] if ap_ is not None else None
            row[f'normal_freq_E{i}'] = nf_[i] if nf_ is not None else None
            row[f'anomaly_freq_E{i}'] = af_[i] if af_ is not None else None
        rows_data.append(row)

    fpath_xlsx = os.path.join(save_path, 'exp2_normal_vs_anomaly_data.xlsx')
    pd.DataFrame(rows_data).to_excel(fpath_xlsx, index=False)
    print(f'  Data saved: {fpath_xlsx}')


# ======================================================================
# Experiment 3: Training Routing Evolution from Log Files
# ======================================================================
def parse_log_routing(log_path):
    """Parse P:[...] from log file to get routing probability evolution."""
    pattern = re.compile(r'iter \[(\d+)/(\d+)\].*P:\[([\d.,\s]+)\]')
    iters = []
    probs_list = []
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                it = int(m.group(1))
                probs = [float(x.strip()) for x in m.group(3).split(',')]
                iters.append(it)
                probs_list.append(probs)

    if not iters:
        print(f'  WARNING: No routing data found in {log_path}')
        return None, None
    return np.array(iters), np.array(probs_list)


def parse_log_usage(log_path):
    """Parse usage:[E0:x.xx, ...] from log file."""
    pattern = re.compile(r'iter \[(\d+)/(\d+)\].*usage:\[(.*?)\]')
    iters = []
    usage_list = []
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                it = int(m.group(1))
                usage_str = m.group(3)
                usages = [float(x.split(':')[1]) for x in usage_str.split(',')]
                iters.append(it)
                usage_list.append(usages)
    if not iters:
        return None, None
    return np.array(iters), np.array(usage_list)


def _load_routing_data(path):
    """Load routing data from either log.txt or routing_history.npz."""
    if path is None or not os.path.exists(path):
        return None, None

    if path.endswith('.npz'):
        data = np.load(path)
        return data['iters'], data['probs']
    else:
        return parse_log_routing(path)


def plot_routing_evolution(log_fp=None, log_ce=None, save_path='.'):
    """Experiment 3: Compare f·P vs CE routing evolution over training.
    Accepts either log.txt or routing_history.npz files."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA', '#00ACC1', '#6D4C41', '#546E7A']

    for ax_idx, (log_path, label) in enumerate([(log_fp, 'f·P (legacy)'), (log_ce, 'CE (entropy)')]):
        ax = axes[ax_idx]
        iters, probs = _load_routing_data(log_path)

        if iters is None:
            display = log_path if log_path else 'Not provided'
            ax.text(0.5, 0.5, f'No data\n{display}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(label)
            continue

        num_experts = probs.shape[1]
        for e in range(num_experts):
            ax.plot(iters, probs[:, e], color=colors[e % len(colors)],
                    label=f'Expert {e}', linewidth=1.5, alpha=0.85)

        uniform_val = 1.0 / num_experts
        ax.axhline(y=uniform_val, color='gray', linestyle='--', alpha=0.5,
                   label=f'Uniform ({uniform_val:.2f})')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Avg Router Probability', fontsize=10)
        ax.set_title(f'{label}: Router P Evolution', fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.3)

        std_per_iter = probs.std(axis=1)
        print(f'  {label}: mean std={std_per_iter.mean():.4f}, '
              f'final std={std_per_iter[-1]:.4f}')

    plt.tight_layout()
    fpath = os.path.join(save_path, 'exp3_routing_evolution.png')
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fpath}')

    # Save to Excel
    fpath_xlsx = os.path.join(save_path, 'exp3_routing_evolution_data.xlsx')
    with pd.ExcelWriter(fpath_xlsx, engine='openpyxl') as writer:
        for log_path, sheet_name in [(log_fp, 'fp'), (log_ce, 'ce')]:
            iters_i, probs_i = _load_routing_data(log_path)
            if iters_i is not None:
                n_e = probs_i.shape[1]
                df = pd.DataFrame(probs_i, columns=[f'E{i}' for i in range(n_e)])
                df.insert(0, 'iter', iters_i)
                df['std'] = probs_i.std(axis=1)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f'  Data saved: {fpath_xlsx}')


# ======================================================================
# Experiment 4: Per-Expert Anomaly Map Decomposition
# ======================================================================
def plot_expert_anomaly_maps(model, data_path, categories, device, save_path,
                             batch_size=1, image_size=512, crop_size=448,
                             num_vis=3):
    """
    Experiment 4: For selected categories, show anomaly maps from each expert
    individually and the fused result.
    Layout per row: [Input] [Expert 0] [Expert 1] ... [Expert K] [Fused] [GT]
    """
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    num_experts = model.bottleneck.num_experts

    for cat in categories:
        test_path = os.path.join(data_path, cat)
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
        loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                             shuffle=False, num_workers=0)

        vis_count = 0
        rows = []

        with torch.no_grad():
            for img, gt, label, img_path in loader:
                if label.item() == 0:
                    continue
                if vis_count >= num_vis:
                    break

                img = img.to(device)
                en, de_per_expert, de_fused, router_info = model.forward_decomposed(img)

                fused_amap, _ = cal_anomaly_maps(en, de_fused, img.shape[-1])
                fused_amap = gaussian_kernel(fused_amap)

                expert_amaps = []
                for e_idx in range(num_experts):
                    e_amap, _ = cal_anomaly_maps(en, de_per_expert[e_idx], img.shape[-1])
                    e_amap = gaussian_kernel(e_amap)
                    expert_amaps.append(e_amap)

                probs = router_info['router_probs'][0].cpu().numpy()
                topk_idx = router_info['topk_indices'][0].cpu().numpy()

                rows.append({
                    'img': img[0].cpu(),
                    'gt': gt[0].cpu(),
                    'expert_amaps': [a[0, 0].cpu().numpy() for a in expert_amaps],
                    'fused_amap': fused_amap[0, 0].cpu().numpy(),
                    'probs': probs,
                    'topk': topk_idx,
                })
                vis_count += 1

        if not rows:
            print(f'  {cat}: No anomaly samples found, skipping.')
            continue

        ncols = 2 + num_experts + 2
        nrows = len(rows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
        if nrows == 1:
            axes = axes[np.newaxis, :]

        for r, row_data in enumerate(rows):
            inp_img = row_data['img'].permute(1, 2, 0).numpy()
            inp_img = (inp_img - inp_img.min()) / (inp_img.max() - inp_img.min() + 1e-8)

            axes[r, 0].imshow(inp_img)
            axes[r, 0].set_title('Input', fontsize=8)
            axes[r, 0].axis('off')

            vmax = max(a.max() for a in row_data['expert_amaps'] + [row_data['fused_amap']])
            vmin = 0

            for e_idx in range(num_experts):
                ax = axes[r, 1 + e_idx]
                amap = row_data['expert_amaps'][e_idx]
                ax.imshow(amap, cmap='jet', vmin=vmin, vmax=vmax)
                selected = '★' if e_idx in row_data['topk'] else ''
                ax.set_title(f'E{e_idx} p={row_data["probs"][e_idx]:.2f}{selected}',
                             fontsize=7, color='red' if selected else 'black')
                ax.axis('off')

            ax_fused = axes[r, 1 + num_experts]
            ax_fused.imshow(row_data['fused_amap'], cmap='jet', vmin=vmin, vmax=vmax)
            ax_fused.set_title('Fused', fontsize=8, fontweight='bold')
            ax_fused.axis('off')

            ax_gt = axes[r, 1 + num_experts + 1]
            gt_np = row_data['gt']
            if gt_np.dim() == 3:
                gt_np = gt_np[0]
            ax_gt.imshow(gt_np.numpy(), cmap='gray')
            ax_gt.set_title('GT', fontsize=8)
            ax_gt.axis('off')

        fig.suptitle(f'Expert Anomaly Map Decomposition - {cat}', fontsize=12, y=1.02)
        plt.tight_layout()
        fpath = os.path.join(save_path, f'exp4_expert_maps_{cat}.png')
        plt.savefig(fpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'Saved: {fpath}')


# ======================================================================
# Experiment 5 (Bonus): Router Weight Cosine Similarity Matrix
# ======================================================================
def plot_router_weight_similarity(model, save_path):
    """Visualize cosine similarity between router output-layer weight vectors."""
    W = model.bottleneck.router[-1].weight.detach().cpu()  # [num_experts, hidden]
    W_norm = F.normalize(W, p=2, dim=-1)
    sim = torch.matmul(W_norm, W_norm.t()).numpy()

    num_experts = sim.shape[0]
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(num_experts):
        for j in range(num_experts):
            ax.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(num_experts))
    ax.set_xticklabels([f'E{i}' for i in range(num_experts)])
    ax.set_yticklabels([f'E{i}' for i in range(num_experts)])
    ax.set_title('Router Weight Cosine Similarity', fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fpath = os.path.join(save_path, 'exp5_router_weight_similarity.png')
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fpath}')

    off_diag = sim[~np.eye(num_experts, dtype=bool)]
    print(f'  Off-diagonal similarity: mean={off_diag.mean():.4f}, max={off_diag.max():.4f}')
    print(f'  -> Lower = router projections more orthogonal = better expert distinction')

    # Save to Excel
    ecols = [f'E{i}' for i in range(num_experts)]
    fpath_xlsx = os.path.join(save_path, 'exp5_router_weight_similarity_data.xlsx')
    pd.DataFrame(sim, index=ecols, columns=ecols).to_excel(fpath_xlsx)
    print(f'  Data saved: {fpath_xlsx}')


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Router Effectiveness Visualization')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['heatmap', 'evolution', 'expert_maps', 'all'],
                        help='Which visualization to generate')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained MoE model checkpoint')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--routing_loss', type=str, default='entropy',
                        choices=['entropy', 'legacy'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path', type=str, default='./vis_results')

    parser.add_argument('--log_fp', type=str, default=None,
                        help='Log file from f*P training (for evolution plot)')
    parser.add_argument('--log_ce', type=str, default=None,
                        help='Log file from CE training (for evolution plot)')

    parser.add_argument('--vis_categories', nargs='+', default=None,
                        help='Categories for expert_maps mode (default: 3 representative ones)')

    args = parser.parse_args()

    if args.data_path is None:
        if args.dataset == 'mvtec':
            args.data_path = '../mvtec_anomaly_detection'
        else:
            args.data_path = '../VisA_pytorch/1cls'

    item_list = MVTEC_ITEMS if args.dataset == 'mvtec' else VISA_ITEMS
    os.makedirs(args.save_path, exist_ok=True)

    need_model = args.mode in ('heatmap', 'expert_maps', 'all')

    model = None
    if need_model:
        assert args.model_path is not None, '--model_path required for this mode'
        model = build_model(args, args.device)

    print('\n' + '=' * 60)

    if args.mode in ('heatmap', 'all'):
        print('\n[Exp 1+2] Collecting routing statistics...')
        stats = collect_routing_stats(model, args.data_path, item_list, args.device)

        print('\n[Exp 1] Category-Expert Routing Preference Heatmap')
        plot_category_expert_heatmap(stats, item_list, args.save_path, top_k=args.top_k)

        print('\n[Exp 2] Normal vs Anomaly Routing Difference')
        plot_normal_vs_anomaly(stats, item_list, args.save_path)

        print('\n[Exp 5] Router Weight Similarity')
        plot_router_weight_similarity(model, args.save_path)

    if args.mode in ('evolution', 'all'):
        print('\n[Exp 3] Routing Evolution Comparison')
        plot_routing_evolution(args.log_fp, args.log_ce, args.save_path)

    if args.mode in ('expert_maps', 'all'):
        vis_cats = args.vis_categories
        if vis_cats is None:
            if args.dataset == 'mvtec':
                vis_cats = ['carpet', 'bottle', 'transistor']
            else:
                vis_cats = ['candle', 'pcb1', 'cashew']
        print(f'\n[Exp 4] Expert Anomaly Map Decomposition: {vis_cats}')
        plot_expert_anomaly_maps(model, args.data_path, vis_cats, args.device, args.save_path)

    print('\n' + '=' * 60)
    print(f'All results saved to: {args.save_path}')
    print('Done!')


if __name__ == '__main__':
    main()
