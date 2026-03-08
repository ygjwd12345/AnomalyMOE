import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math

from models.vision_transformer import bMlp


class MoEBottleneck(nn.Module):
    """
    Mixture-of-Experts Bottleneck: 多个 bMlp 作为 experts，
    通过 MLP router 动态选择 top_k 个 expert 并加权组合输出。
    支持两种路由约束策略：
      - 'legacy': 传统 f·P load balancing + output diversity loss
      - 'entropy': 信息论路由约束 (local sharpness + global balance + orthogonality)
    """
    def __init__(self, dim, hidden_dim=None, num_experts=4, top_k=2,
                 prompt_k=16, dropout=0.2,
                 routing_loss='entropy',
                 lb_weight=0.01, diversity_weight=0.0,
                 lambda_local=0.1, lambda_global=0.5, lambda_ortho=0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.num_experts = num_experts
        self.top_k = top_k
        self.prompt_k = prompt_k
        self.routing_loss = routing_loss

        self.lb_weight = lb_weight
        self.diversity_weight = diversity_weight

        self.lambda_local = lambda_local
        self.lambda_global = lambda_global
        self.lambda_ortho = lambda_ortho

        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_experts),
        )

        self.experts = nn.ModuleList([
            bMlp(dim, self.hidden_dim, dim, drop=dropout)
            for _ in range(num_experts)
        ])

    def compute_entropy_routing_loss(self, router_probs):
        """
        Entropy-based routing constraints (Mutual Information Maximization).
        router_probs: [B, num_experts]

        Returns:
          local_entropy_loss:  minimize per-sample entropy -> sharp routing
          global_entropy_loss: maximize batch-level entropy -> uniform usage
          ortho_loss:          keep router weights orthogonal -> distinct experts
        """
        # 1. Local Sharpness: minimize per-sample entropy H(p_i)
        local_entropy_loss = torch.sum(
            -router_probs * torch.log(router_probs + 1e-6), dim=-1
        ).mean()

        # 2. Global Balance: maximize batch-mean entropy -> minimize negative entropy
        mean_probs = router_probs.mean(dim=0)
        global_entropy_loss = torch.sum(
            mean_probs * torch.log(mean_probs + 1e-6)
        )

        # 3. Orthogonality: keep router output-layer weights distinct
        expert_weights = self.router[-1].weight  # [num_experts, dim//4]
        weights_norm = F.normalize(expert_weights, p=2, dim=-1)
        sim_matrix = torch.matmul(weights_norm, weights_norm.t())
        identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        ortho_loss = torch.norm(sim_matrix - identity, p='fro')

        return local_entropy_loss, global_entropy_loss, ortho_loss

    def forward(self, x):
        B, N, C = x.shape

        # Prompt selection -> context
        x_mean = x.mean(dim=1, keepdim=True)
        sim = F.cosine_similarity(x, x_mean, dim=-1)
        _, topk_idx = torch.topk(sim, min(self.prompt_k, N), dim=1)
        prompts = torch.gather(x, 1, topk_idx.unsqueeze(-1).expand(-1, -1, C))
        context = prompts.mean(dim=1)

        # Routing
        router_logits = self.router(context)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K selection
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Expert outputs + weighted combination
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        expert_mask = torch.zeros(B, self.num_experts, device=x.device)
        expert_mask.scatter_(1, topk_indices, 1)

        indices_expanded = topk_indices.unsqueeze(1).unsqueeze(-1).expand(B, N, self.top_k, C)
        selected_outputs = torch.gather(expert_outputs, 2, indices_expanded)

        weights_expanded = topk_weights.unsqueeze(1).unsqueeze(-1)
        output = (selected_outputs * weights_expanded).sum(dim=2)

        # Scatter renormalized topk_weights back to full expert dimension
        topk_weights_full = torch.zeros(B, self.num_experts, device=x.device)
        topk_weights_full.scatter_(1, topk_indices, topk_weights)

        # --- Routing loss computation ---
        if self.routing_loss == 'entropy':
            local_ent, global_ent, ortho = self.compute_entropy_routing_loss(router_probs)
            router_info = {
                'local_ent_loss': local_ent,
                'global_ent_loss': global_ent,
                'ortho_loss': ortho,
                'expert_usage': expert_mask.mean(dim=0),
                'router_probs_mean': router_probs.mean(dim=0).detach(),
                'router_probs': router_probs.detach(),
                'topk_indices': topk_indices.detach(),
                'topk_weights': topk_weights_full.detach(),
            }
        else:
            # Legacy: f·P load balancing + output diversity
            f = expert_mask.float().mean(dim=0)
            P = router_probs.mean(dim=0)
            lb_loss = self.num_experts * (f * P).sum() * self.lb_weight

            diversity_loss = torch.tensor(0.0, device=x.device)
            if self.diversity_weight > 0 and self.num_experts > 1:
                expert_repr = expert_outputs.mean(dim=1)
                expert_repr = F.normalize(expert_repr, dim=-1)
                sim_mat = torch.bmm(expert_repr, expert_repr.transpose(1, 2))
                eye = torch.eye(self.num_experts, device=x.device).unsqueeze(0)
                off_diag = sim_mat * (1 - eye)
                diversity_loss = off_diag.pow(2).sum() / (B * self.num_experts * (self.num_experts - 1))
                diversity_loss = diversity_loss * self.diversity_weight

            router_info = {
                'lb_loss': lb_loss,
                'diversity_loss': diversity_loss,
                'expert_usage': expert_mask.mean(dim=0),
                'router_probs': router_probs.detach(),
                'topk_indices': topk_indices.detach(),
                'topk_weights': topk_weights_full.detach(),
            }

        return output, router_info

    def forward_single_expert(self, x, expert_idx):
        """Run only a single expert (for visualization / decomposition)."""
        return self.experts[expert_idx](x)


class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.use_moe = isinstance(bottleneck, MoEBottleneck)

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):

        en_list = self.encoder.get_intermediate_layers(x, n=self.target_layers, norm=False)

        side = int(math.sqrt(en_list[0].shape[1]))

        x = self.fuse_feature(en_list)

        router_info = None
        if self.use_moe:
            x, router_info = self.bottleneck(x)
        else:
            for i, blk in enumerate(self.bottleneck):
                x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        if router_info is not None:
            return en, de, router_info
        return en, de


    def forward_decomposed(self, x):
        """
        Per-expert anomaly map decomposition.
        Returns:
            en: list of encoder feature maps (same as normal forward)
            de_per_expert: list of [de_map_for_expert_0, de_map_for_expert_1, ...]
            de_fused: fused decoder output (normal routing)
            router_info: routing info dict with per-sample router_probs
        """
        assert self.use_moe, "forward_decomposed only works with MoE bottleneck"

        en_list = self.encoder.get_intermediate_layers(x, n=self.target_layers, norm=False)
        side = int(math.sqrt(en_list[0].shape[1]))
        feat = self.fuse_feature(en_list)

        fused_output, router_info = self.bottleneck(feat)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_per_expert = []
        for expert_idx in range(self.bottleneck.num_experts):
            expert_out = self.bottleneck.forward_single_expert(feat, expert_idx)
            z = expert_out
            de_list_e = []
            for blk in self.decoder:
                z = blk(z, attn_mask=attn_mask)
                de_list_e.append(z)
            de_list_e = de_list_e[::-1]
            de_e = [self.fuse_feature([de_list_e[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
            de_e = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de_e]
            de_per_expert.append(de_e)

        z = fused_output
        de_list_fused = []
        for blk in self.decoder:
            z = blk(z, attn_mask=attn_mask)
            de_list_fused.append(z)
        de_list_fused = de_list_fused[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de_fused = [self.fuse_feature([de_list_fused[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de_fused = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de_fused]

        return en, de_per_expert, de_fused, router_info

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class ViTillCat(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[1, 3, 5, 7],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTillCat, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        for i, blk in enumerate(self.decoder):
            x = blk(x)

        en = [torch.cat([en_list[idx] for idx in self.fuse_layer_encoder], dim=2)]
        de = [x]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

class ViTAD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 5, 8, 11],
            fuse_layer_encoder=[0, 1, 2],
            fuse_layer_decoder=[2, 5, 8],
            mask_neighbor_size=0,
            remove_class_token=False,
    ) -> None:
        super(ViTAD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = x[:, 1 + self.encoder.num_register_tokens:, :]

        # x = torch.cat(en_list, dim=2)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [en_list[idx] for idx in self.fuse_layer_encoder]
        de = [de_list[idx] for idx in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de


class ViTillv2(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7]
    ) -> None:
        super(ViTillv2, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en.append(x)

        x = self.fuse_feature(en)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de.append(x)

        side = int(math.sqrt(x.shape[1]))

        en = [e[:, self.encoder.num_register_tokens + 1:, :] for e in en]
        de = [d[:, self.encoder.num_register_tokens + 1:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en[::-1], de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTillv3(nn.Module):
    def __init__(
            self,
            teacher,
            student,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_dropout=0.,
    ) -> None:
        super(ViTillv3, self).__init__()
        self.teacher = teacher
        self.student = student
        if fuse_dropout > 0:
            self.fuse_dropout = nn.Dropout(fuse_dropout)
        else:
            self.fuse_dropout = nn.Identity()
        self.target_layers = target_layers
        if not hasattr(self.teacher, 'num_register_tokens'):
            self.teacher.num_register_tokens = 0

    def forward(self, x):
        with torch.no_grad():
            patch = self.teacher.prepare_tokens(x)
            x = patch
            en = []
            for i, blk in enumerate(self.teacher.blocks):
                if i <= self.target_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en.append(x)
            en = self.fuse_feature(en, fuse_dropout=False)

        x = patch
        de = []
        for i, blk in enumerate(self.student):
            x = blk(x)
            if i in self.target_layers:
                de.append(x)
        de = self.fuse_feature(de, fuse_dropout=False)

        en = en[:, 1 + self.teacher.num_register_tokens:, :]
        de = de[:, 1 + self.teacher.num_register_tokens:, :]
        side = int(math.sqrt(en.shape[1]))

        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        return [en.contiguous()], [de.contiguous()]

    def fuse_feature(self, feat_list, fuse_dropout=False):
        if fuse_dropout:
            feat = torch.stack(feat_list, dim=1)
            feat = self.fuse_dropout(feat).mean(dim=1)
            return feat
        else:
            return torch.stack(feat_list, dim=1).mean(dim=1)


class ReContrast(nn.Module):
    def __init__(
            self,
            encoder,
            encoder_freeze,
            bottleneck,
            decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return en_freeze + en, de

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
