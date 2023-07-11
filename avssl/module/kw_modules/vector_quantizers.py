# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import Fp32GroupNorm



class SimpleVectorQuantizer(nn.Module):
    def __init__(
        self,
        temp,
        groundTruthPerplexity=None,
        time_first=True,
        use_gumbel=False,
        hard=True,
        ent_temp=0.01,
    ):
        super().__init__()
        self.time_first = time_first
        self.use_gumbel = use_gumbel
        self.hard = hard
        self.ent_temp = ent_temp

        if isinstance(temp, str):
            import ast

            if temp.startswith("learnable="):
                self.temp_type = "learnable"
                temp = temp.replace("learnable=", "")
                temp = ast.literal_eval(temp)
                self.curr_temp = nn.parameter.Parameter(torch.FloatTensor([temp]))
                logger.info("Setting vq temp learnable (init={})".format(temp))
            elif temp.startswith("fixed="):
                self.temp_type = "fixed"
                temp = temp.replace("fixed=", "")
                temp = ast.literal_eval(temp)
                self.register_buffer("curr_temp", torch.FloatTensor([temp]))
                # self.curr_temp = torch.FloatTensor([temp])
                logger.info("Setting vq temp fixed={}".format(temp))
            else:
                self.temp_type = "scheduled"
                temp = ast.literal_eval(temp)
                assert len(temp) == 3, f"{temp}, {len(temp)}"

                self.max_temp, self.min_temp, self.temp_decay = temp
                logger.info("Setting vq temp scheduled = ({},{},{})".format(*temp))
                self.curr_temp = self.max_temp
        self.codebook_indices = None

        self.groundTruthPerplexity = groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            self.perplexity_criteria = nn.MSELoss()

    def set_num_updates(self, num_updates):
        if self.temp_type == "scheduled":
            self.curr_temp = max(
                self.max_temp * self.temp_decay**num_updates, self.min_temp
            )

    def forward(self, x, prob_msk=[0, 2, 3], feat_len=None, produce_targets=True):

        if not self.time_first:
            x = x.transpose(1, 2)

        result = {"num_vars": x.shape[-1]}
        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz) # (bsz, tsz, grps * num_vars)
        x = x.view(bsz * tsz * 1, -1) # (bsz * tsz * grps, num_vars)

        if feat_len is not None:
            start = 0
            for _l, start in zip(feat_len, range(0, bsz * tsz, tsz)):
                for i in prob_msk:
                    # exclude special token
                    x[start : start + _l, i] += float("-inf")
                # include pad token
                x[start + _l : start + tsz, 0] += float("inf")
        else:
            # exclude special token
            for i in prob_msk:
                x[:, i] += float("-inf")

        with torch.no_grad():
            _, k = x.max(-1) # k is the indices of the largest logits among num_vars

            # hard_x: one hot for the choosen codewords ( bsz * tsz, num_vars )
            hard_x = (
                x.new_zeros(*x.shape)
                .scatter_(-1, k.view(-1, 1), 1.0)
                .view(bsz * tsz, 1, -1)
                .squeeze()
            )
            
            # hard_probs: probs for all codewords in each codebook group : (grp, num_vars) (use one-hot as prob)
            hard_probs = torch.mean(hard_x.float(), dim=0)

            # codebook perplexity sigma {e^(entropy for codebook group)} for all codebook groups
            result["code_perplexity"] = torch.exp(
                -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
            ).sum()

        # average over minibatch and all timesteps of the codewords logits and get their prob by softmax (grp, num_vars)
        avg_probs = torch.softmax(x.view(bsz * tsz, 1, -1).float(), dim=-1).mean(dim=0)

        probs_per_t = (
            torch.softmax(x.view(bsz, tsz, -1), dim=-1).contiguous().permute(1, 0, 2)
        ) # (tsz, bsz, num_vars)

        assert probs_per_t.shape[0] == tsz
        assert probs_per_t.shape[1] == bsz

        ent_per_t = -torch.sum(probs_per_t * torch.log(probs_per_t + 1e-9), dim=-1) # (tsz,bsz)
        ent_per_t = ent_per_t.mean(dim=-1) # (tsz,)
        del probs_per_t
        
        result["ent_per_t"] = ent_per_t

        # prob_cpx : probs for all codewords in each codebook group : (grp, num_vars) (use softmax as prob)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp.item()
        if self.training:
            if self.use_gumbel:
                x = F.gumbel_softmax(
                    x.float(), tau=self.curr_temp, hard=self.hard
                ).type_as(x)
            else:
                x = x / self.curr_temp
                x = F.softmax(x, dim=-1).type_as(x)
                if self.hard:
                    x = hard_x + x - x.detach()
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1) # (bsz * tsz, group * num_vars)

        # add gumbel softmax hard target
        result["subword_prob"] = x.view(bsz, tsz, -1)
        
        # if groundTruthPerplexity is given, minimized the l2 norm with groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            result["diversity_loss"] = (
                self.perplexity_criteria(
                    result["prob_perplexity"],
                    torch.tensor(self.groundTruthPerplexity).type_as(x),
                )
                / (result["num_vars"] - self.groundTruthPerplexity) ** 2
            )
        else:
            result["diversity_loss"] = (
                result["num_vars"] - result["prob_perplexity"]
            ) / result["num_vars"]
            result["diversity_loss"] = result["diversity_loss"] / self.ent_temp
        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * 1, -1).argmax(dim=-1).view(bsz, tsz, 1).detach()
            )

        return result
    
class GumbelVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        num_vars,
        temp,
        groups,
        combine_groups,
        vq_dim,
        time_first,
        activation=nn.GELU(),
        weight_proj_depth=1,
        weight_proj_factor=1,
        init_codebook=None,
        groundTruthPerplexity=None,
    ):
        """Vector quantization using gumbel softmax

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            temp: temperature for training. this should be a tuple of 3 elements: (start, stop, decay factor)
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            activation: what activation to use (should be a module). this is only used if weight_proj_depth is > 1
            weight_proj_depth: number of layers (with activation in between) to project input before computing logits
            weight_proj_factor: this is used only if weight_proj_depth is > 1. scales the inner dimensionality of
                                projections by this factor
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.time_first = time_first
        self.num_vars = num_vars

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        if init_codebook is not None:
            if isinstance(init_codebook, torch.Tensor):
                # init self.vars with init_codebook
                vq_dim = init_codebook.size(-1)
                num_vars = init_codebook.size(0)
                self.vars = nn.Parameter(
                    init_codebook.view(1, num_groups * num_vars, var_dim)
                )
            elif init_codebook == 0:
                # no codebook needed
                self.vars = None
            else:
                raise NotImplementedError()
        else:
            self.vars = nn.Parameter(
                torch.FloatTensor(1, num_groups * num_vars, var_dim)
            )
            nn.init.uniform_(self.vars)

        if weight_proj_depth > 1:

            def block(input_dim, output_dim):
                return nn.Sequential(nn.Linear(input_dim, output_dim), activation)

            inner_dim = self.input_dim * weight_proj_factor
            self.weight_proj = nn.Sequential(
                *[
                    block(self.input_dim if i == 0 else inner_dim, inner_dim)
                    for i in range(weight_proj_depth - 1)
                ],
                nn.Linear(inner_dim, groups * num_vars),
            )
        else:
            self.weight_proj = nn.Linear(self.input_dim, groups * num_vars)
            nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
            nn.init.zeros_(self.weight_proj.bias)

        if isinstance(temp, str):
            import ast

            temp = ast.literal_eval(temp)
        assert len(temp) == 3, f"{temp}, {len(temp)}"

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp
        self.codebook_indices = None

        self.groundTruthPerplexity = groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            self.perplexity_criteria = nn.MSELoss()

    def set_num_updates(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay**num_updates, self.min_temp
        )

    def get_codebook_indices(self):
        if self.codebook_indices is None:
            from itertools import product

            p = [range(self.num_vars)] * self.groups
            inds = list(product(*p))
            self.codebook_indices = torch.tensor(
                inds, dtype=torch.long, device=self.vars.device
            ).flatten()

            if not self.combine_groups:
                self.codebook_indices = self.codebook_indices.view(
                    self.num_vars**self.groups, -1
                )
                for b in range(1, self.groups):
                    self.codebook_indices[:, b] += self.num_vars * b
                self.codebook_indices = self.codebook_indices.flatten()
        return self.codebook_indices

    def codebook(self):
        indices = self.get_codebook_indices()
        return (
            self.vars.squeeze(0)
            .index_select(0, indices)
            .view(self.num_vars**self.groups, -1)
        )

    def sample_from_codebook(self, b, n):
        indices = self.get_codebook_indices()
        indices = indices.view(-1, self.groups)
        cb_size = indices.size(0)
        assert (
            n < cb_size
        ), f"sample size {n} is greater than size of codebook {cb_size}"
        sample_idx = torch.randint(low=0, high=cb_size, size=(b * n,))
        indices = indices[sample_idx]

        z = self.vars.squeeze(0).index_select(0, indices.flatten()).view(b, n, -1)
        return z

    def to_codebook_index(self, indices):
        res = indices.new_full(indices.shape[:-1], 0)
        for i in range(self.groups):
            exponent = self.groups - i - 1
            res += indices[..., i] * (self.num_vars**exponent)
        return res

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars * self.groups}

        if not self.time_first:
            x = x.transpose(1, 2)

        bsz, tsz, fsz = x.shape
        x = x.reshape(-1, fsz)
        x = self.weight_proj(x)
        # x.shape = (bsz, tsz, grps * num_vars)

        x = x.view(bsz * tsz * self.groups, -1)
        # x.shape = (bsz * tsz * grps, num_vars)

        # k is the indices of the largest logits among num_vars
        _, k = x.max(-1)

        # hard_x: one hot for the choosen codewords ( bsz * tsz, self.groups, num_vars )
        hard_x = (
            x.new_zeros(*x.shape)
            .scatter_(-1, k.view(-1, 1), 1.0)
            .view(bsz * tsz, self.groups, -1)
        )

        # hard_probs: probs for all codewords in each codebook group : (grp, num_vars) (use one-hot as prob)
        hard_probs = torch.mean(hard_x.float(), dim=0)

        # codebook perplexity sigma {e^(entropy for codebook group)} for all codebook groups
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        # average over minibatch and all timesteps of the codewords logits and get their prob by softmax (grp, num_vars)
        avg_probs = torch.softmax(
            x.view(bsz * tsz, self.groups, -1).float(), dim=-1
        ).mean(dim=0)

        # prob_cpx : probs for all codewords in each codebook group : (grp, num_vars) (use softmax as prob)
        result["prob_perplexity"] = torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1)
        ).sum()

        result["temp"] = self.curr_temp

        if self.training:
            x = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x = hard_x

        x = x.view(bsz * tsz, -1)
        # x (bsz * tsz, group * num_vars)

        # add gumbel softmax hard target
        result["subword_prob"] = x.view(bsz, tsz, -1)

        # if groundTruthPerplexity is given, minimized the l2 norm with groundTruthPerplexity
        if self.groundTruthPerplexity is not None:
            result["loss"] = (
                self.perplexity_criteria(
                    result["prob_perplexity"],
                    torch.tensor(self.groundTruthPerplexity).type_as(x),
                )
                / (result["num_vars"] - self.groundTruthPerplexity) ** 2
            )
        else:
            result["loss"] = (result["num_vars"] - result["prob_perplexity"]) / result[
                "num_vars"
            ]

        if produce_targets:
            result["targets"] = (
                x.view(bsz * tsz * self.groups, -1)
                .argmax(dim=-1)
                .view(bsz, tsz, self.groups)
                .detach()
            )

        vars = self.vars
        if vars is not None:
            # calculate the following only if codebook exists
            if self.combine_groups:
                # codebook groups shared same set of parameters
                vars = vars.repeat(1, self.groups, 1)

            x = x.unsqueeze(-1) * vars
            # print(x.dtype)
            x = x.view(bsz * tsz, self.groups, self.num_vars, -1)
            x = x.sum(-2).type_as(x)
            x = x.view(bsz, tsz, -1)

            if not self.time_first:
                x = x.transpose(1, 2)  # BTC -> BCT

            result["x"] = x

        return result


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