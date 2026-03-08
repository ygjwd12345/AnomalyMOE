# MoE Bottleneck 版本 - 基于官方 dinomaly_mvtec_uni_base.py
# 唯一区别：bottleneck 从单个 bMlp 替换为 MoEBottleneck

import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.uad import ViTill, ViTillv2, MoEBottleneck
from models import vit_encoder
from dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2, ConvBlock, FeatureJitter
from dataset import MVTecDataset
import torch.backends.cudnn as cudnn
import argparse
from utils import evaluation_batch, global_cosine, regional_cosine_hm_percent, global_cosine_hm_percent, \
    WarmCosineScheduler
from torch.nn import functional as F
from functools import partial
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

from dinov3.hub.backbones import load_dinov3_model

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(item_list):
    setup_seed(1)

    total_iters = 50000
    batch_size = 16
    image_size = 512
    crop_size = 448

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []

    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
        train_data_list.append(train_data)
        test_data_list.append(test_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
                                                   drop_last=True)


    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder_name = 'dinov3_vitb16'
    encoder_weight = 'weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'

    encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers, pretrained_weight_path=encoder_weight)

    if 'vits' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'vitb' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'vitl' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise "Architecture not in vits, vitb, vitl."

    # ===== MoE Bottleneck (Entropy-Based Routing) =====
    bottleneck = MoEBottleneck(
        dim=embed_dim,
        hidden_dim=embed_dim * 4,
        num_experts=4,
        top_k=3,
        prompt_k=16,
        dropout=0.2,
        routing_loss='entropy',
        lambda_local=0.5,
        lambda_global=0.1,
        lambda_ortho=0,
    )

    decoder = []
    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder, fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)

    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Router 初始化: weight=0, bias=-5
    nn.init.constant_(bottleneck.router[2].weight, 0)
    nn.init.constant_(bottleneck.router[2].bias, -5)

    print_fn('=' * 60)
    print_fn('MoE Config:')
    print_fn(f'  num_experts={bottleneck.num_experts}, top_k={bottleneck.top_k}, '
             f'prompt_k={bottleneck.prompt_k}')
    print_fn(f'  hidden_dim={bottleneck.hidden_dim}, dropout={bottleneck.experts[0].drop.p}')
    print_fn(f'  routing_loss={bottleneck.routing_loss}')
    if bottleneck.routing_loss == 'entropy':
        print_fn(f'  lambda_local={bottleneck.lambda_local}, lambda_global={bottleneck.lambda_global}, '
                 f'lambda_ortho={bottleneck.lambda_ortho}')
    else:
        print_fn(f'  lb_weight={bottleneck.lb_weight}, diversity_weight={bottleneck.diversity_weight}')
    print_fn(f'  router_init: weight=0, bias=-5')
    print_fn(f'  total_iters={total_iters}, amsgrad=False')
    print_fn('=' * 60)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=False, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)

    print_fn('train image number:{}'.format(len(train_data)))

    routing_history = {'iters': [], 'probs': [], 'usage': []}

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        model.encoder.eval()

        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)
            en, de, router_info = model(img)

            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            recon_loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            if bottleneck.routing_loss == 'entropy':
                local_ent = router_info['local_ent_loss']
                global_ent = router_info['global_ent_loss']
                ortho = router_info['ortho_loss']
                aux_loss = (bottleneck.lambda_local * local_ent
                            + bottleneck.lambda_global * global_ent
                            + bottleneck.lambda_ortho * ortho)
            else:
                aux_loss = router_info['lb_loss'] + router_info['diversity_loss']

            loss = recon_loss + aux_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)

            optimizer.step()
            loss_list.append(recon_loss.item())
            lr_scheduler.step()

            if (it + 1) % 5000 == 0:

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                  num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)

                    print_fn(
                        '{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn(
                    'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))

                model.train()
                model.encoder.eval()

            it += 1
            if it == total_iters:
                break

        expert_usage = router_info['expert_usage'].cpu().numpy()
        usage_str = ', '.join([f'E{i}:{u:.2f}' for i, u in enumerate(expert_usage)])
        if bottleneck.routing_loss == 'entropy':
            probs_mean = router_info['router_probs_mean'].cpu().numpy()
            probs_str = ', '.join([f'{p:.3f}' for p in probs_mean])
            print_fn('iter [{}/{}], loss:{:.4f}, L_ent:{:.4f}, G_ent:{:.4f}, ortho:{:.4f}, '
                     'P:[{}], usage:[{}]'.format(
                it, total_iters, np.mean(loss_list),
                local_ent.item(), global_ent.item(), ortho.item(),
                probs_str, usage_str))
        else:
            print_fn('iter [{}/{}], loss:{:.4f}, lb:{:.4f}, div:{:.4f}, usage:[{}]'.format(
                it, total_iters, np.mean(loss_list),
                router_info['lb_loss'].item(), router_info['diversity_loss'].item(), usage_str))

        routing_history['iters'].append(it)
        routing_history['probs'].append(router_info['router_probs_mean'].cpu().numpy()
                                        if 'router_probs_mean' in router_info
                                        else router_info['expert_usage'].cpu().numpy())
        routing_history['usage'].append(expert_usage)

    torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))
    np.savez(os.path.join(args.save_dir, args.save_name, 'routing_history.npz'),
             iters=np.array(routing_history['iters']),
             probs=np.array(routing_history['probs']),
             usage=np.array(routing_history['usage']))

    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='mvtec_uni_dinov3_moe2')
    args = parser.parse_args()

    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(device)
    train(item_list)
