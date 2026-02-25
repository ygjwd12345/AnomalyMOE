"""
测试已保存模型的性能。
支持 baseline (bMlp bottleneck) 和 MoE bottleneck 两种模型。

用法:
  # 测试 MVTec baseline
  python test_model.py --dataset mvtec --model_path ./saved_results/vitill_mvtec_uni_dinov3_base/model.pth

  # 测试 MVTec MoE
  python test_model.py --dataset mvtec --model_path ./saved_results/mvtec_uni_dinov3_moe2/model.pth --use_moe --num_experts 2 --top_k 1

  # 测试 VisA baseline
  python test_model.py --dataset visa --model_path ./saved_results/visa_uni_dinov3_base/model.pth

  # 测试 VisA MoE
  python test_model.py --dataset visa --model_path ./saved_results/visa_uni_dinov3_moe2/model.pth --use_moe --num_experts 2 --top_k 1
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import warnings
from functools import partial

from models.uad import ViTill, MoEBottleneck
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from dataset import get_data_transforms, MVTecDataset
from utils import evaluation_batch
from dinov3.hub.backbones import load_dinov3_model

warnings.filterwarnings("ignore")

MVTEC_ITEMS = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
               'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

VISA_ITEMS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
              'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


def build_model(args, device):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder_name = 'dinov3_vitb16'
    encoder_weight = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers,
                                pretrained_weight_path=encoder_weight)

    embed_dim, num_heads = 768, 12

    if args.use_moe:
        bottleneck = MoEBottleneck(
            dim=embed_dim,
            hidden_dim=embed_dim * 4,
            num_experts=args.num_experts,
            top_k=args.top_k,
            prompt_k=16,
            dropout=0.2,
            lb_weight=0.01,
        )
    else:
        bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.2)])

    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                 qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                 attn=LinearAttention2)
        for _ in range(8)
    ])

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=target_layers, mask_neighbor_size=0,
                   fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)

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


def test(args):
    device = args.device
    batch_size = 16
    image_size = 512
    crop_size = 448

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    if args.dataset == 'mvtec':
        item_list = MVTEC_ITEMS
    elif args.dataset == 'visa':
        item_list = VISA_ITEMS
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    model = build_model(args, device)

    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for item in item_list:
        test_path = os.path.join(args.data_path, item)
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                      shuffle=False, num_workers=4)

        results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        print('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

    print('=' * 80)
    print('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test saved model')
    parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'visa'])
    parser.add_argument('--data_path', type=str, default=None,
                        help='Dataset root path. Default: ../mvtec_anomaly_detection or ../VisA_pytorch/1cls')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model.pth')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--use_moe', action='store_true', help='Use MoE bottleneck')
    parser.add_argument('--num_experts', type=int, default=2)
    parser.add_argument('--top_k', type=int, default=1)

    args = parser.parse_args()

    if args.data_path is None:
        if args.dataset == 'mvtec':
            args.data_path = '../mvtec_anomaly_detection'
        else:
            args.data_path = '../VisA_pytorch/1cls'

    test(args)
