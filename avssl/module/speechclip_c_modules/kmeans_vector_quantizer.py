# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from fairseq.modules import Fp32GroupNorm


class KmeansVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        gamma=0.25,
        init_codebook=None,
    ):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()
        if init_codebook is not None:
            if isinstance(init_codebook, torch.nn.Embedding):
                vq_dim = init_codebook.embedding_dim
                num_vars = init_codebook.num_embeddings
            else:
                vq_dim = init_codebook.size(-1)
                num_vars = init_codebook.size(0)

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1
        self.num_groups = num_groups

        if init_codebook is not None:
            # init_codebook = init_codebook.view(
            #     num_vars, num_groups, self.var_dim
            # ).detach()
            self.embedding = init_codebook
        else:
            self.embedding = nn.Parameter(
                0.01 * torch.randn(num_vars, num_groups, self.var_dim)
            )
        self.projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False),
            Fp32GroupNorm(groups, dim),
        )
        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction="mean")

    def _pass_grad(self, x, y):
        """Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """

        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if self.combine_groups:
            if isinstance(self.embedding, torch.nn.Embedding):
                return self.embedding.weight.view(
                    self.num_vars, self.num_groups, self.var_dim
                ).expand(self.num_vars, self.groups, self.var_dim)
            else:
                return self.embedding.expand(self.num_vars, self.groups, self.var_dim)
        return self.embedding.weight.view(self.num_vars, self.num_groups, self.var_dim)

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars}

        if self.time_first:
            x = x.transpose(1, 2)

        bsz, fsz, tsz = x.shape

        ze = self.projection(x)
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        d = (
            (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1))
            .view(self.num_vars, bsz, tsz, self.groups, -1)
            .norm(dim=-1, p=2)
        )

        prob_proj = nn.Softmax(dim=-1)
        result["subword_prob"] = prob_proj(-1 * d.squeeze(-1).permute(1, 2, 0))

        idx = d.argmin(dim=0)
        zq = (
            torch.stack(
                [
                    self.expand_embedding[idx[..., group], group]
                    for group in range(self.groups)
                ],
                dim=-2,
            )
            .view(bsz, tsz, self.groups * self.var_dim)
            .permute(0, 2, 1)
        )
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)

        hard_x = (
            idx.new_zeros(bsz * tsz * self.groups, self.num_vars)
            .scatter_(-1, idx.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)

        result["code_cpx"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        if produce_targets:
            result["targets"] = idx

        if self.time_first:
            x = x.transpose(1, 2)  # BCT -> BTC
        result["x"] = x

        ze = ze.float()
        zq = zq.float()
        # latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())

        # result["loss"] = latent_loss + self.gamma * commitment_loss
        result["loss"] = self.gamma * commitment_loss
        return result
