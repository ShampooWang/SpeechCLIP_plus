import logging
import math
from mailbox import NotEmptyError
from typing import List, Tuple, Union

import numpy
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

MAX_FEAT_LEN = 75

# source: https://github.com/MingLunHan/CIF-PyTorch/blob/main/modules/cif_middleware.py
class CifMiddleware(nn.Module):
    def __init__(
        self,
        cal_quantity_loss=True,
        using_gt_len=True,
        cif_threshold=1.0,
        cif_embedding_dim=768,
        encoder_embed_dim=768,
        produce_weight_type="conv",
        conv_cif_width=3,
        conv_cif_dropout=0.1,
        apply_scaling=True,
        apply_tail_handling=True,
        tail_handling_firing_threshold=0.5,
        **cfg,
    ):
        super().__init__()

        # Load configurations
        self.cal_quantity_loss = cal_quantity_loss
        self.using_gt_len = using_gt_len
        self.cif_threshold = cif_threshold
        self.cif_output_dim = cif_embedding_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.produce_weight_type = produce_weight_type
        self.conv_cif_width = conv_cif_width
        self.conv_cif_dropout = conv_cif_dropout
        self.apply_scaling = apply_scaling
        self.apply_tail_handling = apply_tail_handling
        self.tail_handling_firing_threshold = tail_handling_firing_threshold

        # Build weight generator
        if self.produce_weight_type == "dense":
            self.dense_proj = Linear(
                self.encoder_embed_dim, self.encoder_embed_dim
            ).cuda()
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()
        elif self.produce_weight_type == "conv":
            self.conv = torch.nn.Conv1d(
                self.encoder_embed_dim,
                self.encoder_embed_dim,
                self.conv_cif_width,
                stride=1,
                padding=int(self.conv_cif_width / 2),
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ).cuda()
            self.conv_dropout = torch.nn.Dropout(p=self.conv_cif_dropout).cuda()
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()
        else:
            self.weight_proj = Linear(self.encoder_embed_dim, 1).cuda()

        # Build the final projection layer (if encoder_embed_dim is not equal to cif_output_dim)
        if self.cif_output_dim != self.encoder_embed_dim:
            self.cif_output_proj = Linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False
            ).cuda()

    def forward(self, encoder_outputs, target_lengths):
        """
        Args:
            encoder_outputs: a dictionary that includes
                encoder_raw_out:
                    the raw outputs of acoustic encoder, with shape B x T x C
                encoder_padding_mask:
                    the padding mask (whose padded regions are filled with ones) of encoder outputs, with shape B x T
            target_lengths: the length of targets (necessary when training), with shape B
        Return:
            A dictionary:
                cif_out:
                    the cif outputs
                cif_out_padding_mask:
                    the padding infomation for cif outputs (whose padded regions are filled with zeros)
                quantity_out:
                    the sum of weights for the calculation of quantity loss
        """

        # Collect inputs
        encoder_raw_outputs = encoder_outputs["encoder_raw_out"]  # B x T x C
        encoder_padding_mask = encoder_outputs["encoder_padding_mask"]  # B x T

        # Produce weights for integration (accumulation)
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(encoder_raw_outputs)
            act_proj_out = torch.relu(proj_out)
            sig_input = self.weight_proj(act_proj_out)
            weight = torch.sigmoid(sig_input)
        elif self.produce_weight_type == "conv":
            conv_input = encoder_raw_outputs.permute(0, 2, 1)
            conv_out = self.conv(conv_input)
            proj_input = conv_out.permute(0, 2, 1)
            proj_input = self.conv_dropout(proj_input)
            sig_input = self.weight_proj(proj_input)
            weight = torch.sigmoid(sig_input)
        else:
            sig_input = self.weight_proj(encoder_raw_outputs)
            weight = torch.sigmoid(sig_input)
        # weight has shape B x T x 1

        not_padding_mask = ~encoder_padding_mask
        weight = (
            torch.squeeze(weight, dim=-1) * not_padding_mask.int()
        )  # weight has shape B x T
        org_weight = weight

        # Apply scaling strategies
        if self.training and self.apply_scaling and target_lengths is not None:
            # Conduct scaling when training
            weight_sum = weight.sum(-1)  # weight_sum has shape B
            normalize_scalar = torch.unsqueeze(
                target_lengths / weight_sum, -1
            )  # normalize_scalar has shape B x 1
            weight = weight * normalize_scalar

        # Prepare for Integrate and fire
        batch_size = encoder_raw_outputs.size(0)
        max_length = encoder_raw_outputs.size(1)
        encoder_embed_dim = encoder_raw_outputs.size(2)
        padding_start_id = not_padding_mask.sum(-1)  # shape B

        accumulated_weights = torch.zeros(batch_size, 0).cuda()
        accumulated_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()
        fired_states = torch.zeros(batch_size, 0, encoder_embed_dim).cuda()

        # Begin integrate and fire
        for i in range(max_length):
            # Get previous states from the recorded tensor
            prev_accumulated_weight = (
                torch.zeros([batch_size]).cuda()
                if i == 0
                else accumulated_weights[:, i - 1]
            )
            prev_accumulated_state = (
                torch.zeros([batch_size, encoder_embed_dim]).cuda()
                if i == 0
                else accumulated_states[:, i - 1, :]
            )

            # Decide whether to fire a boundary
            cur_is_fired = (
                (prev_accumulated_weight + weight[:, i]) >= self.cif_threshold
            ).unsqueeze(dim=-1)
            # cur_is_fired with shape B x 1

            # Update the accumulated weights
            cur_weight = torch.unsqueeze(weight[:, i], -1)
            # cur_weight has shape B x 1
            prev_accumulated_weight = torch.unsqueeze(prev_accumulated_weight, -1)
            # prev_accumulated_weight also has shape B x 1
            remained_weight = (
                torch.ones_like(prev_accumulated_weight).cuda()
                - prev_accumulated_weight
            )
            # remained_weight with shape B x 1

            # Obtain the accumulated weight of current step
            cur_accumulated_weight = torch.where(
                cur_is_fired,
                cur_weight - remained_weight,
                cur_weight + prev_accumulated_weight,
            )  # B x 1

            # Obtain accumulated state of current step
            cur_accumulated_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                (cur_weight - remained_weight) * encoder_raw_outputs[:, i, :],
                prev_accumulated_state + cur_weight * encoder_raw_outputs[:, i, :],
            )  # B x C

            # Obtain fired state of current step:
            # firing locations has meaningful representations, while non-firing locations is all-zero embeddings
            cur_fired_state = torch.where(
                cur_is_fired.repeat(1, encoder_embed_dim),
                prev_accumulated_state + remained_weight * encoder_raw_outputs[:, i, :],
                torch.zeros([batch_size, encoder_embed_dim]).cuda(),
            )  # B x C

            # Handle the tail
            if (not self.training) and self.apply_tail_handling:
                # When encoder output position exceeds the max valid position,
                # if accumulated weights is greater than tail_handling_firing_threshold,
                # current state should be reserved, otherwise it is discarded.
                cur_fired_state = torch.where(
                    i
                    == padding_start_id.unsqueeze(dim=-1).repeat(
                        [1, encoder_embed_dim]
                    ),
                    # shape B x C
                    torch.where(
                        cur_accumulated_weight.repeat([1, encoder_embed_dim])
                        <= self.tail_handling_firing_threshold,
                        # shape B x C
                        torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                        # less equal than tail_handling_firing_threshold, discarded.
                        cur_accumulated_state / (cur_accumulated_weight + 1e-10)
                        # bigger than tail_handling_firing_threshold, normalized and kept.
                    ),
                    cur_fired_state,
                )
                # shape B x T

            # For normal condition, including both training and evaluation
            # Mask padded locations with all-zero vectors
            cur_fired_state = torch.where(
                torch.full([batch_size, encoder_embed_dim], i).cuda()
                > padding_start_id.unsqueeze(dim=-1).repeat([1, encoder_embed_dim]),
                torch.zeros([batch_size, encoder_embed_dim]).cuda(),
                cur_fired_state,
            )

            # Update accumulation-related values: T_c stands for the length of integrated features
            accumulated_weights = torch.cat(
                (accumulated_weights, cur_accumulated_weight), 1
            )  # B x T_c
            accumulated_states = torch.cat(
                (accumulated_states, torch.unsqueeze(cur_accumulated_state, 1)), 1
            )  # B x T_c x C
            fired_states = torch.cat(
                (fired_states, torch.unsqueeze(cur_fired_state, 1)), 1
            )  # B x T_c x C

        # Extract cif_outputs for each utterance
        fired_marks = (torch.abs(fired_states).sum(-1) != 0.0).int()  # B x T_c
        fired_utt_length = fired_marks.sum(-1)  # B
        fired_max_length = max(
            1, min(fired_utt_length.max().int(), MAX_FEAT_LEN)
        )  # The maximum of fired times in current batch
        cif_outputs = torch.zeros([0, fired_max_length, encoder_embed_dim]).cuda()

        def dynamic_partition(
            data: torch.Tensor, partitions: torch.Tensor, num_partitions=None
        ):
            assert (
                len(partitions.shape) == 1
            ), "Only one dimensional partitions supported"
            assert (
                data.shape[0] == partitions.shape[0]
            ), "Partitions requires the same size as data"
            if num_partitions is None:
                num_partitions = max(torch.unique(partitions))
            return [data[partitions == index] for index in range(num_partitions)]

        # Loop over all samples
        cif_outputs_len = []
        for j in range(batch_size):
            # Get information of j-th sample
            cur_utt_fired_mark = fired_marks[j, :]
            cur_utt_fired_state = fired_states[j, :, :]
            cur_utt_outputs = dynamic_partition(
                cur_utt_fired_state, cur_utt_fired_mark, 2
            )
            cur_utt_output = cur_utt_outputs[1]  # Get integrated representations
            cur_utt_length = cur_utt_output.size(0)  # The total number of firing

            # handle empty output
            # if cur_utt_length == 0:
            #     cur_utt_output = accumulated_states[j, -1, :].unsqueeze(0)
            #     cur_utt_length = cur_utt_output.size(0)

            if cur_utt_length > MAX_FEAT_LEN:
                assert cur_utt_output.dim() == 2, cur_utt_output.shape
                cur_utt_output = cur_utt_output[:MAX_FEAT_LEN, :]
                cur_utt_length = cur_utt_output.size(0)

            assert (
                cur_utt_length <= fired_max_length
            ), f"out: {cur_utt_length}, max: {fired_max_length}"
            pad_length = fired_max_length - cur_utt_length  # Get padded length
            cif_outputs_len.append(cur_utt_length)
            cur_utt_output = torch.cat(
                (
                    cur_utt_output,
                    torch.full([pad_length, encoder_embed_dim], 0.0).cuda(),
                ),
                dim=0,
            )  # Pad current utterance cif outputs to fired_max_length
            cur_utt_output = torch.unsqueeze(cur_utt_output, 0)
            # Reshape to 1 x T_c x C

            # Concatenate cur_utt_output and cif_outputs along batch axis
            cif_outputs = torch.cat([cif_outputs, cur_utt_output], 0)

        cif_out_padding_mask = (torch.abs(cif_outputs).sum(-1) != 0.0).int()
        # cif_out_padding_mask has shape B x T_c, where locations with value 0 are the padded locations.

        if self.training:
            quantity_out = org_weight.sum(-1)
        else:
            quantity_out = weight.sum(-1)

        if self.cif_output_dim != encoder_embed_dim:
            cif_outputs = self.cif_output_proj(cif_outputs)

        cif_output_len_diff = [
            abs(_x - _y) for _x, _y in zip(cif_outputs_len, target_lengths)
        ]

        cif_outputs_len = [min(max(1, _l), MAX_FEAT_LEN) for _l in cif_outputs_len]
        cif_outputs_len = torch.LongTensor(cif_outputs_len)
        cif_outputs_len = cif_outputs_len.to(cif_outputs.device)

        return {
            "cif_out": cif_outputs,  # B x T_c x C
            "cif_out_padding_mask": cif_out_padding_mask,  # B x T_c
            "quantity_out": quantity_out,  # B
            "cif_outputs_len": cif_outputs_len,  # B
            "target_len": target_lengths,  # B
            "cif_weight": weight,  # B x T
            "cif_output_len_diff": cif_output_len_diff,  # B
            "fired_marks": fired_marks,  # B x T x C
        }


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class AlphaNetwork(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.feat_dim,
                out_channels=self.feat_dim,
                kernel_size=5,
                stride=1,
                padding="same",
                dilation=1,
                groups=1,
                bias=True,
            ),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(self.feat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # bsz, seq_len, hid_dim = x.shape
        # return torch.ones(bsz, seq_len,1).type_as(x) * 1e-5
        assert x.shape[2] == self.feat_dim
        x = x.permute(0, 2, 1)
        # x (bsz, hid_dim, seq_len,)
        x = self.conv_layer(x)
        x = x.permute(0, 2, 1)
        # x (bsz, seq_len, hid_dim)
        x = self.output_layer(x)
        return x


class CIF(nn.Module):
    def __init__(
        self,
        audio_feat_dim,
        beta=1.0,
        scaling_stragety=False,
        cal_quantity_loss=False,
        tail_handling=False,
    ):
        super().__init__()
        self.audio_feat_dim = audio_feat_dim

        # threshold (default=1.0)
        self.beta = beta
        self.weight_network = AlphaNetwork(feat_dim=self.audio_feat_dim)

        self.scaling_stragety = scaling_stragety
        self.cal_quantity_loss = cal_quantity_loss
        self.tail_handling = tail_handling

        if self.cal_quantity_loss:
            self.quantity_loss_criteria = nn.L1Loss()

    def forward(
        self, encoder_outputs, encoder_lens=None, target_length=None, paddingTensor=None
    ):
        device = encoder_outputs.device
        # encoder_outputs = (bsz,seq_len,hid_dim)
        assert encoder_outputs.shape[2] == self.audio_feat_dim
        bsz, seq_len = encoder_outputs.shape[:2]

        if encoder_lens is not None:
            assert encoder_lens.shape[0] == bsz

        alphas = self.weight_network(encoder_outputs)
        assert alphas.shape == (bsz, seq_len, 1), alphas.shape

        if encoder_lens is not None:
            enc_output_msk = (
                torch.arange(seq_len)[None, :].to(device)
                < encoder_lens.view(bsz, 1)[:, None]
            ).squeeze()
        else:
            enc_output_msk = torch.ones((bsz, seq_len)).to(device)

        if self.cal_quantity_loss:
            assert (
                target_length is not None
            ), "target_length cannot be None to calculate quantity_loss"
            quantity_loss = self.quantity_loss_criteria(
                torch.sum(alphas.view(bsz, seq_len) * enc_output_msk, dim=-1),
                target_length,
            )

        if self.training:
            # training
            if target_length is not None and self.scaling_stragety:
                # scaling strategy
                assert target_length.shape[0] == bsz

                alphas = alphas.view(bsz, seq_len)
                assert alphas.shape == enc_output_msk.shape

                alphas = (
                    alphas
                    / torch.sum(alphas * enc_output_msk, dim=-1, keepdim=True)
                    * target_length.view(bsz, 1)
                )
                alphas = alphas.view(bsz, seq_len, 1)

        output_c = []
        max_c_len = 0
        for bsz_i in range(bsz):
            alpha = alphas[bsz_i, :, :]
            h = encoder_outputs[bsz_i, :, :]
            alpha_accum = [torch.zeros((1,)).to(device)]
            h_accum = [torch.zeros((self.audio_feat_dim,)).to(device)]
            c = []

            assert alpha.shape == (seq_len, 1)
            assert h.shape == (seq_len, self.audio_feat_dim)

            for u in range(1, seq_len + 1):
                if encoder_lens is not None:
                    if u > encoder_lens[bsz_i]:
                        break
                # u : current timestep (start from 1 to seq_len)
                alpha_u = alpha[u - 1]
                h_u = h[u - 1, :]

                # alpha^a_u = alpha^a_(u-1) + alpha_u
                alpha_accum_u = alpha_accum[u - 1] + alpha_u
                alpha_accum.append(alpha_accum_u)

                assert len(alpha_accum)

                if alpha_accum_u < self.beta:
                    # no boundary is located
                    # h^a_u = h^a_(u-1) + alpha_u * h_u
                    h_accum.append(h_accum[u - 1] + alpha_u * h_u)
                else:
                    # boundart located
                    # divide alpha into 2 parts : alpha_u1 and alpha_u2

                    alpha_u1 = 1 - alpha_accum[u - 1]

                    c.append(h_accum[u - 1] + alpha_u1 * h_u)

                    alpha_u2 = alpha_u - alpha_u1
                    alpha_accum_u = alpha_u2
                    alpha_accum[-1] = alpha_accum_u
                    h_accum.append(alpha_u2 * h_u)

            if self.tail_handling and not self.training:
                # add additional firing if residual weight > 0.5
                if alpha_accum[-1] > 0.5:
                    c.append(h_accum[-1])
            # handle empty c
            if len(c) == 0:
                c = [h_accum[-1]]
            c = torch.stack(c)
            max_c_len = max(max_c_len, c.shape[0])
            output_c.append(c)

        if paddingTensor is None:
            paddingTensor = torch.zeros(
                self.audio_feat_dim,
            ).to(device)

        len_tensor = []
        for i in range(len(output_c)):
            assert max_c_len >= output_c[i].shape[0], "{} {}".format(
                max_c_len, output_c[i].shape[0]
            )
            len_tensor.append((output_c[i].shape[0]))
            output_c[i] = torch.cat(
                [
                    output_c[i],
                    paddingTensor.view(1, self.audio_feat_dim).repeat(
                        max_c_len - output_c[i].shape[0], 1
                    ),
                ],
                dim=0,
            )
            assert output_c[i].shape == (max_c_len, self.audio_feat_dim)

        output_c = torch.stack(output_c, dim=0)

        assert output_c.shape == (bsz, max_c_len, self.audio_feat_dim)
        len_tensor = torch.tensor(len_tensor).to(device)

        if self.cal_quantity_loss:
            return output_c, len_tensor, quantity_loss
        else:
            return output_c, len_tensor


class CIF_Forced(nn.Module):
    def __init__(
        self,
        audio_feat_dim: int,
        beta: float = 1.0,
        output_length: int = 8,
    ):
        super().__init__()
        self.audio_feat_dim = audio_feat_dim

        # threshold (default=1.0)
        self.beta = beta
        self.weight_network = AlphaNetwork(feat_dim=self.audio_feat_dim)
        self.output_length = output_length

    def forward(
        self,
        encoder_outputs,
        encoder_lens=None,
        cal_q_loss: bool = False,
        returnAlphas: bool = False,
    ):
        device = encoder_outputs.device
        # encoder_outputs = (bsz,seq_len,hid_dim)
        assert encoder_outputs.shape[2] == self.audio_feat_dim
        bsz, seq_len = encoder_outputs.shape[:2]

        if encoder_lens is not None:
            assert encoder_lens.shape[0] == bsz
            for x in encoder_lens:
                assert x >= self.output_length, (x, self.output_length)

        alphas = self.weight_network(encoder_outputs)
        assert alphas.shape == (bsz, seq_len, 1), alphas.shape

        if encoder_lens is not None:
            enc_output_msk = (
                (
                    torch.arange(seq_len)[None, :].to(device)
                    < encoder_lens.view(bsz, 1)[:, None]
                )
                .squeeze()
                .bool()
            )
        else:
            enc_output_msk = torch.ones((bsz, seq_len)).bool().to(device)

        # if self.cal_quantity_loss:
        #     assert (
        #         target_length is not None
        #     ), "target_length cannot be None to calculate quantity_loss"
        #     quantity_loss = self.quantity_loss_criteria(
        #         torch.sum(alphas.view(bsz, seq_len) * enc_output_msk, dim=-1),
        #         target_length,
        #     )

        # force scaling to desired output length

        alphas = alphas.view(bsz, seq_len)
        assert alphas.shape == enc_output_msk.shape

        # original_alphas = alphas

        if cal_q_loss:
            q_loss = F.mse_loss(
                torch.sum(alphas * enc_output_msk, dim=-1),
                torch.FloatTensor([self.output_length for _ in range(bsz)]).type_as(
                    alphas
                ),
            )

        alphas = (
            alphas
            / torch.sum(alphas * enc_output_msk, dim=-1, keepdim=True)
            * self.output_length
        )

        alphas[~enc_output_msk] = 0.0

        alphas = alphas.view(bsz, seq_len, 1)

        output_c = []
        max_c_len = 0
        for bsz_i in range(bsz):
            alpha = alphas[bsz_i, :, :]
            h = encoder_outputs[bsz_i, :, :]
            alpha_accum = [torch.zeros((1,)).to(device)]
            h_accum = [torch.zeros((self.audio_feat_dim,)).to(device)]
            c = []

            assert alpha.shape == (seq_len, 1)
            assert h.shape == (seq_len, self.audio_feat_dim)

            for u in range(1, seq_len + 1):
                # print("u=",u)
                if encoder_lens is not None:
                    if u > encoder_lens[bsz_i]:
                        break
                # u : current timestep (start from 1 to seq_len)
                alpha_u = alpha[u - 1]
                h_u = h[u - 1, :]
                # assert alpha_u <= 1.0, (alpha_u,encoder_lens[bsz_i],alpha,original_alphas[bsz_i])
                # print("currrent alpha",alpha_u.item())

                # alpha^a_u = alpha^a_(u-1) + alpha_u
                alpha_accum_u = alpha_accum[u - 1] + alpha_u
                alpha_accum.append(alpha_accum_u)

                assert len(alpha_accum)

                # print("alpha_accum_u",alpha_accum_u.item())

                if alpha_accum_u < self.beta - 1e-5:
                    # no boundary is located
                    # h^a_u = h^a_(u-1) + alpha_u * h_u
                    h_accum.append(h_accum[u - 1] + alpha_u * h_u)
                else:
                    # boundary located
                    # divide alpha into 2 parts : alpha_u1 and alpha_u2

                    alpha_u1 = 1 - alpha_accum[u - 1]

                    c.append(h_accum[u - 1] + alpha_u1 * h_u)

                    alpha_u2 = alpha_u - alpha_u1
                    while alpha_u2 > self.beta:
                        # alpha_u2 still > beta
                        c.append(self.beta * h_u)
                        alpha_u2 = alpha_u2 - self.beta

                    # print("[Fired] split {} = {} + {}".format(alpha_u.item(),alpha_u1.item(),alpha_u2.item()))
                    alpha_accum_u = alpha_u2
                    alpha_accum[-1] = alpha_accum_u
                    h_accum.append(alpha_u2 * h_u)

            # assert alpha_accum[-1] == 0, alpha_accum[-1]
            if alpha_accum[-1] > 0.999:
                # due to precision problem
                # we must fire here
                c.append(h_accum[-1])

            # print("alpha_accum[-1]",alpha_accum[-1])

            c = torch.stack(c)
            max_c_len = max(max_c_len, c.shape[0])
            output_c.append(c)

        output_c = torch.stack(output_c, dim=0)
        assert output_c.shape == (bsz, self.output_length, self.audio_feat_dim)

        if returnAlphas:
            if cal_q_loss:
                return output_c, q_loss, alphas
            else:
                return output_c, alphas
        else:
            if cal_q_loss:
                return output_c, q_loss
            else:
                return output_c


class CNN(nn.Module):
    def __init__(self, embd_dim, hparams: dict) -> None:
        super().__init__()

        def conv1d_length(
            length: Union[torch.Tensor, list],
            kernel: int,
            stride: int,
            pad: int,
            dilation: int,
        ):
            if isinstance(length, torch.Tensor):
                for i in range(length.size(0)):
                    length[i] = math.floor(
                        (length[i] + 2 * pad - dilation * (kernel - 1)) / stride + 1
                    )
            elif isinstance(length, list):
                for i in range(len(length)):
                    length[i] = math.floor(
                        (length[i] + 2 * pad - dilation * (kernel - 1)) / stride + 1
                    )
            else:
                raise TypeError()

        def mean_pool_length(
            length: Union[torch.Tensor, list], kernel: int, stride: int, pad: int
        ):
            new_length = []
            if isinstance(length, torch.Tensor):
                for i in range(length.size(0)):
                    length[i] = math.floor((length[i] + 2 * pad - kernel) / stride + 1)
            elif isinstance(length, list):
                for i in range(len(length)):
                    length[i] = math.floor((length[i] + 2 * pad - kernel) / stride + 1)
            else:
                raise TypeError()

        self.model_hparams = hparams

        module_list = []
        for _model_name in self.model_hparams:
            _hparams = self.model_hparams[_model_name]
            assert isinstance(
                _hparams, list
            ), f"wrong type of layer params: {type(_hparams)}"

            if "cnn" in _model_name:
                assert len(_hparams) >= 4
                module_list.append(nn.Conv1d(embd_dim, embd_dim, *_hparams))
            elif "mean_pool" in _model_name:
                assert len(_hparams) >= 3
                module_list.append(nn.AvgPool1d(*_hparams))
            else:
                raise NotImplementedError(_model_name)

        self.downsample_modules = nn.ModuleList(module_list)
        logger.info("CNN downsampling module list:\n{}".format(self.downsample_modules))

    def compute_len(
        self,
        downsampling_layer: str,
        audio_len: Union[torch.Tensor, list],
        kernel: int,
        stride: int,
        pad: int,
        dilation: int = 1,
    ):

        new_len = []
        assert isinstance(audio_len, torch.Tensor) | isinstance(
            audio_len, list
        ), f"wrong audio_length type: {type(audio_len)}"
        if "cnn" in downsampling_layer:
            for _l in audio_len:
                _l = math.floor((_l + 2 * pad - dilation * (kernel - 1)) / stride + 1)
                if _l <= 0:
                    _l = 1
                new_len.append(_l)
        elif "mean_pool" in downsampling_layer:
            for _l in audio_len:
                _l = math.floor((_l + 2 * pad - kernel) / stride + 1)
                if _l <= 0:
                    _l = 1
                new_len.append(_l)
        else:
            raise NotImplementedError(downsampling_layer)

        if isinstance(audio_len, torch.Tensor):
            new_len = torch.tensor(new_len)

        return new_len

    def forward(self, audio_feats: torch.Tensor, audio_len: Union[torch.Tensor, list]):
        assert audio_feats.dim() == 3
        assert isinstance(audio_len, torch.Tensor) | isinstance(
            audio_len, list
        ), f"wrong audio_length type: {type(audio_len)}"

        audio_feats = audio_feats.permute(0, 2, 1)
        for m in self.downsample_modules:
            audio_feats = m(audio_feats)
        audio_feats = audio_feats.permute(0, 2, 1)

        for _model_name in self.model_hparams:
            _hparams = self.model_hparams[_model_name]
            audio_len = self.compute_len(_model_name, audio_len, *_hparams)

        if isinstance(audio_len, torch.Tensor):
            audio_len = torch.clamp(audio_len, max=MAX_TEXT_LEN)
        else:
            audio_len = [min(_l, MAX_TEXT_LEN) for _l in audio_len]

        return audio_feats, audio_len


if __name__ == "__main__":
    bsz = 4
    audio_dim = 768
    seq_len = 150
    max_tar_len = 8

    def get_keypadding_mask(max_length: int, data_lens: torch.Tensor) -> torch.Tensor:
        """Create keypadding mask for attention layers

        Args:
            max_length (int): the max sequence length of the batch
            audio_len (torch.Tensor): the lens for each data in the batch, shape = (bsz,)

        Returns:
            torch.Tensor: key_padding_mask, bool Tensor, True for padding
        """
        bsz = data_lens.size(0)
        key_padding_mask = torch.ones([bsz, max_length])
        for mask, len in zip(key_padding_mask, data_lens):
            mask[:len] = 0.0
        key_padding_mask = key_padding_mask.type_as(data_lens).bool()

        return key_padding_mask

    cif = CifMiddleware()
    opt = torch.optim.SGD(cif.parameters(), lr=0.1)

    audio_input = torch.randn(bsz, seq_len, audio_dim)
    audio_input_lens = torch.randint(max_tar_len, seq_len, (bsz,))
    print(audio_input_lens)

    audio_input = audio_input.cuda()
    audio_input_lens = audio_input_lens.cuda()
    encoder_outputs = {
        "encoder_raw_out": audio_input,
        "encoder_padding_mask": get_keypadding_mask(
            audio_input.shape[1], audio_input_lens
        ),
    }
    target_length = torch.randint(1, max_tar_len, (bsz,)).cuda()

    for i in range(10):

        cif = cif.cuda()

        opt.zero_grad()

        # output_c, audio_input_lens1, q_loss = cif(
        #     encoder_outputs=audio_input,
        #     encoder_lens=None,
        #     target_length=audio_input_lens - 2,
        # )

        output = cif(encoder_outputs, target_length)
        for _val in output.values():
            print(_val)
        exit(1)
        q_loss.backward()
        opt.step()

        print("step", i, q_loss.item())
    for x, l, gl in zip(output_c, audio_input_lens1, audio_input_lens):
        print("predict_len,gold_len", l.item(), gl.item())

    # inference
    cif.eval()

    with torch.no_grad():
        audio_input = torch.randn(bsz, seq_len, audio_dim)
        audio_input_lens = torch.randint(4, seq_len, (bsz,))

        audio_input = audio_input.cuda()
        audio_input_lens = audio_input_lens.cuda()
        output_c, audio_input_lens, q_loss = cif(
            encoder_outputs=audio_input,
            encoder_lens=None,
            target_length=audio_input_lens - 2,
        )

    print("Inference")
    print(q_loss)
    for x, l in zip(output_c, audio_input_lens):
        print(x.shape, l)
