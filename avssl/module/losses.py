"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn
from fairseq.criterions import FairseqCriterion


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


MAX_EYE = 256


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


class HybridLoss(nn.Module):
    def __init__(
        self, init_temp=0.05, warmup_step=5000, max_temp=10, criterion=nn.MSELoss()
    ):
        super().__init__()
        self.criterion = getattr(nn, criterion)(reduction="mean")
        self.init_temp = init_temp
        self.temp = init_temp
        self.warmup_step = warmup_step
        self.max_temp = max_temp

    @property
    def current_temperature(self) -> float:
        """Current Temperature

        Returns:
            float: Temperature
        """

        # if self.temperature_trainable:
        #     temp = self.temperature.data.cpu().detach().float().exp().item()
        # else:
        # assert self.temp > 0, f"{self.temp}"
        temp = self.temp

        return float(temp)

    def forward(self, input, target, global_step):
        assert (
            global_step > self.warmup_step
        ), f"current step is {global_step}, but warmup step is {self.warmup_step}"
        loss = self.criterion(input, target)
        loss = loss * self.temp
        self.temp = min(
            self.init_temp + 0.001 * (global_step - self.warmup_step), self.max_temp
        )
        return loss


# source: https://github.com/facebookresearch/fairseq/blob/main/fairseq/criterions/hubert_criterion.py
class HubertCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        pred_masked_weight,
        pred_nomask_weight,
        loss_weights=None,
        log_keys=None,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(target_list=sample["target_list"], **sample["net_input"])
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True)
        targ_m_list = model.get_targets(net_output, True)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[0].numel()

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.item()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = compute_correct(logp_m)
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = compute_correct(logp_u)
                logging_output[f"correct_u_{i}"] = corr_u
                logging_output[f"count_u_{i}"] = count_u

        return loss, sample_size, logging_output


class DiversityLoss(nn.Module):
    def __init__(self, loss_weight=10):
        super().__init__()
        self.loss_weight = loss_weight

    def current_loss_weight(self):
        return self.loss_weight

    def forward(
        self, target=torch.Tensor, target_len=torch.Tensor, score_type=str, eps=1e-8
    ):
        assert target.dim() == 3, target.shape
        score = 0
        for _tar, _len in zip(target, target_len):
            _tar = _tar[:_len]

            if score_type == "corr":
                _score_mtx = torch.corrcoef(_tar)
            elif score_type == "cos":
                _tar_norm = _tar.norm(dim=-1)[:, None]
                _tar = _tar / torch.clamp(_tar_norm, min=eps)
                _score_mtx = _tar @ _tar.T
            else:
                raise NotImplementedError(score_type)

            if _len == 1:
                continue
            else:
                _identity_mtx = torch.eye(_len, device=_score_mtx.device)
                assert _identity_mtx.shape == _score_mtx.shape
                # _score_mtx = torch.square(_score_mtx - _identity_mtx)
            score += torch.sum(_score_mtx - _identity_mtx) / (_len * (_len - 1))

        return score / torch.sum(target_len != 1)
