"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        learnable_temperature=True,
    ):
        super(SupConLoss, self).__init__()
        self.learnable_temperature = learnable_temperature
        if learnable_temperature:
            self.temperature = torch.nn.parameter.Parameter(
                torch.FloatTensor(
                    [
                        temperature,
                    ]
                )
            )
        else:
            self.temperature = temperature

        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    @property
    def current_temperature(self):
        if self.learnable_temperature:
            return self.temperature.item()
        else:
            return self.temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(1 / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


MAX_EYE = 1024


class MaskedContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        temperature_trainable: bool = False,
        margin: float = 0.0,
        dcl: bool = False,
        a2b: bool = True,
        b2a: bool = True,
    ):
        """Masked Contrastive Loss

        Args:
            temperature (float, optional): Temperature for scaling logits. Defaults to 0.07.
            temperature_trainable (bool, optional): Trains temperature. Defaults to False.
            margin (float, optional): Margin. Defaults to 0.0.
            dcl (bool, optional): Decoupled contrastive learning (https://arxiv.org/abs/2110.06848). Defaults to False.
            a2b (bool, optional): Computes A to B classification loss. Defaults to True.
            b2a (bool, optional): Computes B to A classification loss. Defaults to True.
        """

        super().__init__()

        assert a2b or b2a, "Cannot set both `a2b` and `b2a` to False."

        self.temperature_trainable = temperature_trainable
        self.margin = margin
        self.dcl = dcl
        self.a2b = a2b
        self.b2a = b2a

        if temperature_trainable:
            self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            self.temperature = 1 / temperature

        eye_mat = torch.eye(MAX_EYE, dtype=torch.bool)
        logger.info(f"Masked contrastive loss max eye size: {eye_mat.shape}")
        self.register_buffer("eye_mat", eye_mat)
        self.register_buffer("neg_eye_mat", ~eye_mat)
        self.register_buffer("eye_mat_fl", eye_mat.type(torch.float))

    @property
    def current_temperature(self) -> float:
        """Current Temperature

        Returns:
            float: Temperature
        """

        if self.temperature_trainable:
            temp = self.temperature.data.cpu().detach().float().exp().item()
        else:
            temp = self.temperature

        return float(temp)

    def forward(
        self, feat_A: torch.Tensor, feat_B: torch.Tensor, index: torch.LongTensor = None
    ) -> torch.Tensor:
        """Computes loss

        Args:
            feat_A (torch.Tensor): Features from view A or modality A.
            feat_B (torch.Tensor): Features from view B or modality B.
            index (torch.LongTensor, optional): Labels for each sample. Defaults to None.

        Returns:
            torch.Tensor: Loss.
        """

        assert feat_A.shape == feat_B.shape, (feat_A.shape, feat_B.shape)
        B = feat_A.shape[0]

        # Construct masks
        with torch.no_grad():
            if index is not None:
                assert index.shape[0] == feat_A.shape[0], (index.shape, feat_A.shape)
                index = index.unsqueeze(1)
                neg_mask = index != index.t()  # (batch, batch)
            else:
                neg_mask = self.neg_eye_mat[:B, :B].clone()

            pos_mask = self.eye_mat[:B, :B]

            if not self.dcl:
                neg_mask[pos_mask] = True

            neg_mask_fl = neg_mask.type(feat_A.dtype)

        if self.temperature_trainable:
            temperature = torch.exp(self.temperature)
        else:
            temperature = self.temperature

        # Compute logits
        logits = feat_A @ feat_B.t() * temperature

        # Add margin
        if self.margin > 0.0:
            logits[pos_mask] -= self.margin

        # Compute losses
        pos_logits = logits[pos_mask]
        exp_logits = logits.exp() * neg_mask_fl
        loss = 0
        if self.a2b:
            neg_A2B = torch.log(exp_logits.sum(1))
            loss_A2B = (-pos_logits + neg_A2B).mean()
            loss += loss_A2B
        if self.b2a:
            neg_B2A = torch.log(exp_logits.sum(0))
            loss_B2A = (-pos_logits + neg_B2A).mean()
            loss += loss_B2A
        if self.a2b and self.b2a:
            loss = loss / 2

        return loss


class KeywordDiversityLoss(nn.Module):
    def __init__(self, loss_weight=10, keyword_center=None, temp=0.25):
        super().__init__()
        self.loss_weight = loss_weight
        self.temp = temp
        self.keyword_center = keyword_center
        self.criterion = nn.L1Loss()

    def current_loss_weight(self):
        return self.loss_weight

    def forward(
        self, targets=torch.Tensor, target_lens=torch.Tensor, score_type=str, eps=1e-5
    ):
        assert targets.dim() == 3, targets.shape
        assert score_type in ["corr", "cos"], score_type

        if score_type == "corr":
            assert self.keyword_center is not None
            assert not self.keyword_center.requires_grad

        scoreList = []
        for target, target_len in zip(targets, target_lens):
            if target_len == 1:
                continue

            target = target[:target_len]
            if score_type == "corr":
                target = target - self.keyword_center.to(target.device)
            normalize_target = F.normalize(target, dim=-1, eps=eps)
            score_mtx = normalize_target @ normalize_target.T
            identity_mtx = torch.eye(target_len, device=score_mtx.device)
            loss_mask = identity_mtx != 1.0
            assert (
                identity_mtx.shape == score_mtx.shape
            ), f"{identity_mtx.shape}, {score_mtx.shape}"
            scoreList.append(
                self.criterion(
                    (score_mtx / self.temp) * loss_mask,
                    (identity_mtx / self.temp) * loss_mask,
                )
            )

        if len(scoreList) > 0:
            return sum(scoreList) / len(scoreList)
        else:  # Empty
            return 0.0


class AttentionDiversityLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.criterion = nn.L1Loss()
        self.max_Len = round(250000 / 320)

    def current_loss_weight(self):
        return self.loss_weight

    def forward(self, attention_maps, eps=1e-5):
        loss_list = []
        B, L = attention_maps[0].shape[0], attention_maps[0].shape[2]
        target_eye_tensor = (
            torch.eye(attention_maps[0].shape[1], device=attention_maps[0].device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        for attention_map in attention_maps:
            if attention_map.shape[-1] > self.max_Len:
                print(attention_map.shape)
            normalize_attention_map = F.normalize(attention_map, dim=-1, eps=eps)[
                :, :, : self.max_Len, : self.max_Len
            ]
            correlation_score = (
                normalize_attention_map.unsqueeze(2)
                * normalize_attention_map.unsqueeze(1)
            ).sum([-1, -2]) / L
            assert (
                correlation_score.shape == target_eye_tensor.shape
            ), f"{correlation_score.shape}, {target_eye_tensor.shape}"
            loss_list.append(
                self.criterion(correlation_score, target_eye_tensor).mean()
            )

        return sum(loss_list) / len(loss_list)
