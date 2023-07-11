import logging
from multiprocessing.dummy import freeze_support
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ..module import (
    ClipModel,
    MLPLayers,
    AttentionDiversityLoss,
)

from ..module.self_attn_modules import TransformerModels
from ..module.kw_modules import vector_quantizers
from ..module.cif import CIF
from ..module.kw_modules.keyword_batchnorms import Kw_BatchNorm, Kw_BatchNorm_plus

from ..util import (
    get_keypadding_mask, 
    clip_fp_alignment,
)

METRIC_REDUCEFN_MAPPING = {
    torch.Tensor: lambda x: torch.mean(x),
    float: lambda x: x,
    int: lambda x: x,
    str: lambda x: x,
}

logger = logging.getLogger(__name__)


def keyword_extraction_pipeline(model, audio_feat, key_padding_mask, learnable_cls, extract_hiddens=False):
    # Use multi-head attention mechanism to find fixed number of keywords(cls)
    resultDict = { "cascaded_audio_feat": None, "parallel_audio_feat": None, "vq_results": None, "vq_keywords": None }
    bsz = audio_feat.shape[0]
    cls = torch.cat([learnable_cls] * bsz, dim=0)
    src = torch.cat([cls, audio_feat], dim=1)
    
    if extract_hiddens: 
        attn_out, hidden_states = model.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        return hidden_states
    else:
        attn_out, attention_maps = model.self_att.extract_attention_map(
            src=src, key_padding_mask=key_padding_mask
        )
        resultDict["attention_maps"] = attention_maps

    # Parallel branch
    if isinstance(model, ParallelBranch):
        resultDict["parallel_audio_feat"] = model.linear_proj(attn_out[:, 0, :].reshape(-1, model.audio_dim))
        return resultDict
    else:
        # Hybird
        if isinstance(model, HybridBranch):
            resultDict["parallel_audio_feat"] = model.parallel_proj(attn_out[:, 0, :].reshape(-1, model.audio_dim))
            attn_out = attn_out[:, 1 : model.keyword_num+1].reshape(
                -1, model.keyword_num, model.audio_dim
            )
        else:
            attn_out = attn_out[:, : model.keyword_num].reshape(
                -1, model.keyword_num, model.audio_dim
            )
        
    if hasattr(model, "linear_proj"):
        proj_out = model.linear_proj(attn_out)

    if hasattr(model, "bn_layer"):
        bn_proj_out = model.bn_layer(proj_out)

    # Cosine
    if model.keyword_num is not None:
        cos_score = []
        for i in range(model.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    bn_proj_out[:, i, :].view(bsz, model.text_dim, 1),
                    model.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            model.keyword_num,
            model.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, model.keyword_num, model.clip.model.token_embedding.num_embeddings)}"
        
    # VQ
    if hasattr(model, "vector_quantizer"):
        vq_results = model.vector_quantizer(x=cos_score)
        resultDict["vq_results"] = vq_results
        assert model.clip.model.token_embedding.weight.requires_grad == False
        resultDict["vq_keywords"] = vq_results["subword_prob"] @ model.clip.model.token_embedding.weight
        resultDict["cascaded_audio_feat"] = model.clip.encode_keywords(resultDict["vq_keywords"], model.keyword_num)

    return resultDict


class CascadedBranch(nn.Module):
    def __init__(self, config, audio_dim: int, text_dim: int, clip: ClipModel) -> None:
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.clip = clip
        self.config = config
        self.kw_projection_config = (
            self.config.model_settings.cascaded_branch.keyword.get(
                "kw_projection", None
            )
        )

        logger.info("Using CascadedBranch")
        self.keyword_num = getattr(config.model_settings.cascaded_branch.keyword, "number", 8)

        self.cls = self._create_cls()
        logger.info("Start init [CLS] {}".format(self.cls.shape))

        assert hasattr(
            TransformerModels, config.model_settings.cascaded_branch.transformer_args.type
        )
        logger.info(
            f"Using {config.model_settings.cascaded_branch.transformer_args.type} as CascadedBranch"
        )
        self.self_att = getattr(
            TransformerModels, config.model_settings.cascaded_branch.transformer_args.type
        )(**config.model_settings.cascaded_branch.transformer_args)

        if self.kw_projection_config is None:
            logger.info(
                "kw_projection not specified, using single linear layer as default"
            )
            self.linear_proj = nn.Linear(
                self.config.model_settings.cascaded_branch.transformer_args.d_model,
                self.text_dim,
            )
        else:
            logger.info(
                f"kw_projection dims:{self.kw_projection_config.dimensions} droupout:{self.kw_projection_config.dropout}"
            )
            assert (
                self.kw_projection_config.dimensions[0]
                == self.config.model_settings.cascaded_branch.transformer_args.d_model
            ), f"first dim({self.kw_projection_config.dimensions[0]}) should match the audio encoder dim({self.config.model_settings.cascaded_branch.transformer_args.d_model})"
            assert (
                self.kw_projection_config.dimensions[-1] == self.text_dim
            ), f"last dim({self.kw_projection_config.dimensions[-1]}) should match the text encoder dim({self.text_dim})"
            self.linear_proj = MLPLayers(
                units=self.kw_projection_config.dimensions,
                dropout=self.kw_projection_config.dropout,
            )

        # codebook selection
        self.vector_quantizer = None
        self.vq_type = config.model_settings.cascaded_branch.vq.type

        if not hasattr(
            vector_quantizers, config.model_settings.cascaded_branch.vq.type
        ):
            raise NotImplementedError(
                "Vq ({}) not implemented".format(
                    config.model_settings.cascaded_branch.vq.type
                )
            )

        self.vector_quantizer = getattr(vector_quantizers, self.vq_type)(
            **config.model_settings.cascaded_branch.vq.args
        )

        if hasattr(config.model_settings.cascaded_branch.keyword, "batchnorms"):
            self.bn_layer = Kw_BatchNorm(
                kw_num=self.keyword_num,
                kw_dim=self.text_dim,
                batchnorm_type=config.model_settings.cascaded_branch.keyword.batchnorms.type,
                init_bias=torch.mean(self.clip.model.token_embedding.weight, dim=0),
                init_scale=torch.std(self.clip.model.token_embedding.weight, dim=0),
                std_scale=config.model_settings.cascaded_branch.keyword.batchnorms.std_scale,
                learnable=config.model_settings.cascaded_branch.keyword.batchnorms.learnable
                if hasattr(
                    config.model_settings.cascaded_branch.keyword.batchnorms,
                    "learnable",
                )
                else True,
                parallel=config.model_settings.cascaded_branch.keyword.batchnorms.parallel
                if hasattr(
                    config.model_settings.cascaded_branch.keyword.batchnorms, "parallel"
                )
                else False,
            )

    def _create_cls(self):
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    self.keyword_num,
                    self.config.model_settings.cascaded_branch.transformer_args.d_model,
                ]
            )
        )
    
    
    def extract_hidden_states(self, audio_feat, audio_feat_len):
        inputDict = {"audio_feat": audio_feat, "audio_feat_len": audio_feat_len}
        hidden_states = self.forward(inputDict, extract_hiddens=True)

        if hidden_states[0].dim() == 2:
            hidden_states = [ x.unsqueeze(0) for x in hidden_states ]
        
        hidden_states = [ x[:, self.keyword_num:, :] for x in hidden_states ]

        return hidden_states

    def forward(self, inputDict, extract_hiddens=False):
        audio_feat, audio_feat_len = inputDict["audio_feat"], inputDict["audio_feat_len"]
        key_padding_mask = get_keypadding_mask(
            max_length=audio_feat.shape[1] + self.keyword_num,
            data_lens=audio_feat_len + self.keyword_num
        )
        return keyword_extraction_pipeline(self, audio_feat, key_padding_mask, self.cls, extract_hiddens)

    def getAttentionMap(self, audio_feat, audio_len):
        # Use multi-head attention layer to find keywords(cls)
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + self.keyword_num
        )

        _, attn_output_weights = self.self_att.extract_attention_map(
            src=src, key_padding_mask=key_padding_mask
        )

        cls_weights = []
        for i in range(attn_output_weights.shape[0]):
            cls_weights.append(
                attn_output_weights[
                    i, :, : self.keyword_num, : audio_len[i] + self.keyword_num
                ]
            )

        keywords = self.self_att(src=src, key_padding_mask=key_padding_mask)

        keywords = keywords[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.audio_dim
        )

        keywords = self.linear_proj(keywords)

        if hasattr(self, "bn_layer"):
            keywords = self.bn_layer(keywords)

        # cosine
        cos_score = []
        for i in range(self.keyword_num):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )

        cos_score = torch.stack(cos_score, dim=1)
        # disallow special tokens
        cos_score[..., 0] -= 100
        cos_score[..., 2] -= 100
        cos_score[..., 3] -= 100

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        topk_kw = [[[] for _ in range(self.keyword_num)] for _ in range(bsz)]
        _, topk_kw_ids = torch.topk(cos_score, dim=-1, k=10)
        for bsz_i in range(bsz):
            for kw_i in range(self.keyword_num):
                topk_kw[bsz_i][kw_i] = [
                    self.clip.tokenizer.decoder[
                        self.clip.reducedl2Original[x.item()]
                    ].replace("</w>", "")
                    for x in topk_kw_ids[bsz_i, kw_i]
                ]

        return cls_weights, topk_kw, None 


class HybridBranch(CascadedBranch):
    def __init__(
        self, config, audio_dim: int, text_dim: int, out_dim: int, clip: ClipModel
    ) -> None:
        super().__init__(config, audio_dim, text_dim, clip)
        self.out_dim = out_dim
        self.parallel_proj = nn.Linear(self.audio_dim, self.out_dim)

    def _create_cls(self):
        # first cls for parallel objective
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    self.keyword_num + 1,
                    self.config.model_settings.cascaded_branch.transformer_args.d_model,
                ]
            )
        )

    def extract_hidden_states(self, audio_feat, audio_feat_len, use_kw=False):
        hidden_states = self.forward(audio_feat, audio_feat_len, extract_hiddens=True)
        if not use_kw:
            hidden_states = [x[:, self.keyword_num + 1 :, ...] for x in hidden_states]
        else:
            hidden_states = [x[:, 1 : self.keyword_num + 1, ...] for x in hidden_states]

        return tuple(hidden_states)

    def forward(self, inputDict, extract_hiddens=False):
        audio_feat, audio_feat_len = inputDict["audio_feat"], inputDict["audio_feat_len"]
        key_padding_mask = get_keypadding_mask(
            max_length=audio_feat.shape[1] + self.keyword_num + 1,
            data_lens=audio_feat_len + self.keyword_num + 1,
        )
        return keyword_extraction_pipeline(self, audio_feat, key_padding_mask, self.cls, extract_hiddens)


class ParallelBranch(nn.Module):
    def __init__(self, config, audio_dim: int, out_dim: int) -> None:
        super().__init__()
        self.config = config
        self.audio_dim = audio_dim
        self.out_dim = out_dim
        self.need_projection = self.config.model_settings.parallel_branch.get(
            "need_projection", True
        )

        assert hasattr(
            TransformerModels, config.model_settings.parallel_branch.transformer_args.type
        )
        logger.info(
            f"Using {config.model_settings.parallel_branch.transformer_args.type} as ParallelBranch (projection={self.need_projection})"
        )
        self.self_att = getattr(
            TransformerModels, config.model_settings.parallel_branch.transformer_args.type
        )(**config.model_settings.parallel_branch.transformer_args)

        self.cls = self._create_cls()
        logger.info("Start init [CLS] {}".format(self.cls.shape))

        if self.need_projection:
            self.linear_proj = nn.Linear(self.audio_dim, self.out_dim)

    def _create_cls(self):
        # first cls for parallel objective
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    1,
                    self.config.model_settings.parallel_branch.transformer_args.d_model,
                ]
            )
        )

    def extract_hidden_states(self, audio_feat, audio_feat_len):
        inputDict = {"audio_feat": audio_feat, "audio_feat_len": audio_feat_len}
        hidden_states = self.forward(inputDict, extract_hiddens=True)
        
        if hidden_states[0].dim() == 2:
            hidden_states = [ x.unsqueeze(0) for x in hidden_states ]
       
        hidden_states = [ x[:, 1:, :] for x in hidden_states ]

        return hidden_states

    def forward(self, inputDict, extract_hiddens=False):
        audio_feat, audio_feat_len = inputDict["audio_feat"], inputDict["audio_feat_len"]
        key_padding_mask = get_keypadding_mask(
            max_length=audio_feat.shape[1] + 1,
            data_lens=audio_feat_len + 1,
        )
        return keyword_extraction_pipeline(self, audio_feat, key_padding_mask, self.cls, extract_hiddens)


class CascadedBranch_dynamic(nn.Module):
    def __init__(self, config, audio_dim: int, text_dim: int, clip: ClipModel) -> None:
        super().__init__()
        
        logger.info("Using CascadedBranch_plus")
        if hasattr(config.model_settings.cascaded_branch, "transformer_args"):
            self.self_att = getattr(
                TransformerModels, config.model_settings.cascaded_branch.transformer_args.type
            )(**config.model_settings.cascaded_branch.transformer_args)
            logger.info("Using self-attention before downsampling")
            
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.clip = clip
        self.config = config
        self.kw_projection_config = (
            self.config.model_settings.cascaded_branch.keyword.get(
                "kw_projection", None
            )
        )

        if self.kw_projection_config is None:
            logger.info(
                "kw_projection not specified, using single linear layer as default"
            )
            self.linear_proj = nn.Linear(
                self.audio_dim,
                self.text_dim,
            )
        else:
            logger.info(
                f"kw_projection dims:{self.kw_projection_config.dimensions} droupout:{self.kw_projection_config.dropout}"
            )
            assert (
                self.kw_projection_config.dimensions[0] == self.audio_dim
            ), f"first dim({self.kw_projection_config.dimensions[0]}) should match the audio encoder dim({self.audio_dim})"
            assert (
                self.kw_projection_config.dimensions[-1] == self.text_dim
            ), f"last dim({self.kw_projection_config.dimensions[-1]}) should match the text encoder dim({self.text_dim})"

            self.linear_proj = MLPLayers(
                units=self.kw_projection_config.dimensions,
                dropout=self.kw_projection_config.dropout,
            )

        # codebook selection
        self.vector_quantizer = None
        self.vq_type = config.model_settings.cascaded_branch.vq.type

        if not hasattr(
            vector_quantizers, config.model_settings.cascaded_branch.vq.type
        ):
            raise NotImplementedError(
                "Vq ({}) not implemented".format(
                    config.model_settings.cascaded_branch.vq.type
                )
            )

        self.vector_quantizer = getattr(vector_quantizers, self.vq_type)(
            **config.model_settings.cascaded_branch.vq.args
        )

        if hasattr(config.model_settings.cascaded_branch.keyword, "batchnorms"):
            self.bn_layer = Kw_BatchNorm_plus(
                kw_dim=self.text_dim,
                init_bias=torch.mean(self.clip.model.token_embedding.weight, dim=0),
                init_scale=torch.std(self.clip.model.token_embedding.weight, dim=0),
                std_scale=config.model_settings.cascaded_branch.keyword.batchnorms.std_scale,
                learnable=config.model_settings.cascaded_branch.keyword.batchnorms.learnable
                if hasattr(
                    config.model_settings.cascaded_branch.keyword.batchnorms,
                    "learnable",
                )
                else True,
            )

        assert hasattr(config.model_settings.cascaded_branch.downsampling, "type")
        self.downsampling_type = config.model_settings.cascaded_branch.downsampling.type
        if self.downsampling_type == "cif":
            self.using_gt_len = config.model_settings.cascaded_branch.downsampling.get(
                "using_gt_len", False
            )
            if self.using_gt_len:
                logger.info("Using ground truth text length target")
            self.downsampling = CIF(
                **config.model_settings.cascaded_branch.downsampling.cif.__dict__
            )
        else:
            if self.downsampling_type != "true_boundary":
                raise NotImplementedError(
                    "Unknown type:{}".format(config.downsampling.type)
                )
        logger.info("Using {} downsampling method".format(self.downsampling_type))
            
            
    def project_keyword_to_CLIPspace(self, keywords):
        keywords = self.linear_proj(keywords)
        if hasattr(self, "bn_layer"):
            keywords = self.bn_layer(keywords)
            
        return keywords        
            
            
    def get_keyword_cosine_score(self, keywords):
        B = keywords.shape[0]
        N = keywords.shape[1]
        
        cos_score = []
        for i in range(N):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(B, self.text_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
        cos_score = torch.stack(cos_score, dim=1).to(keywords.device)
        
        return cos_score
        
        
    def construct_pseudo_dsample_alpha(self, fp_alignment: torch.Tensor, features: torch.Tensor):
        with torch.no_grad():
            B = fp_alignment.shape[0]
            target_alpha = torch.zeros(features.shape[:2]).to(features.device)  # B x T
            interval_lens = (
                torch.diff(
                    fp_alignment, prepend=torch.zeros([B,1], device=fp_alignment.device), dim=-1
                )
                .clip(min=0)
                .long()
            ).to(features.device)

            for b in range(B):
                pseudo_weight = torch.repeat_interleave(
                    interval_lens[b], interval_lens[b], dim=0
                ).reciprocal()
                interval_l = interval_lens[b].sum().clip(max=target_alpha.shape[-1])
                target_alpha[b, :interval_l] = pseudo_weight[:interval_l]

            return target_alpha
        
        
    def dsample_with_pseudo_weight(self, inputDict: dict, target_len: torch.Tensor):
        clip_fp_alignment(inputDict, inputDict["audio_feat"].shape[1])
        fp_ali = inputDict["fp_alignment"]
        features = inputDict["audio_feat"]
        padding_mask = inputDict["audio_feat_pad_mask"]
        target_len = target_len.clip(max=fp_ali.shape[1])
        
        with torch.no_grad():
            target_alpha = self.construct_pseudo_dsample_alpha(fp_ali, features)
            dsample_feats_length = target_len
            B, T_d, D = (
                target_alpha.shape[0],
                dsample_feats_length.max().long().item(),
                features.shape[-1],
            )

            ##############################
            ## Compute indices to merge ##
            ##############################
            merge_indices = torch.zeros(features.shape[:2], device=features.device)
            interval_lens = (
                torch.diff(
                    fp_ali, prepend=torch.zeros([B,1], device=fp_ali.device), dim=-1
                )
                .clip(min=0)
                .long()
            ).to(features.device)
            position_indices = pad_sequence(
                [
                    torch.arange(seg_num.item(), device=interval_lens.device)
                    for seg_num in target_len
                ],
                batch_first=True,
            )

            for _b in range(B):
                interval_l = interval_lens[_b].sum().clip(max=target_alpha.shape[-1])
                assert position_indices[_b].shape == interval_lens[_b].shape, f'{position_indices[_b].shape}, {interval_lens[_b].shape}'
                merge_indices[_b, :interval_l] = torch.repeat_interleave(
                    position_indices[_b], interval_lens[_b], dim=0
                )[:interval_l]

            fired_marks = torch.cat(
                [merge_indices.diff() > 0, torch.zeros([B,1], device=merge_indices.device)], dim=1
            ).bool()
            merge_indices = merge_indices.unsqueeze(-1).expand(-1, -1, D).long()

        ######################
        ## Merging features ##
        ######################
        weighted_feats = target_alpha.unsqueeze(-1) * features
        dsample_feats = torch.zeros(
            [B, T_d, D], device=merge_indices.device
        ).scatter_add_(1, merge_indices, weighted_feats)

        result_dict = { "dsample_feats": dsample_feats, 
                        "dsample_alpha": target_alpha,
                        "dsample_feats_length": dsample_feats_length, 
                        "dsample_feats_pad_mask": get_keypadding_mask(T_d, dsample_feats_length),
                        "fired_marks": fired_marks, 
                        "target_len": target_len,
                        "input_feats_pad_mask": padding_mask}

        return result_dict
    
    
    def downsampling_audio_feat(self, audio_feat, audio_feat_len, audio_feat_pad_mask, fp_alignment=None, target_len=None):
        inputDict = {"audio_feat": audio_feat, 
                     "audio_feat_len": audio_feat_len, 
                     "audio_feat_pad_mask": audio_feat_pad_mask,}
        
        if self.downsampling_type == "cnn":
            keywords, new_feat_len = self.downsampling(audio_feat, audio_feat_len)
            dsample_results["dsample_feats"] = keywords
            dsample_results["dsample_feats_length"] = new_feat_len
        elif self.downsampling_type == "cif":
            # We don't provide the downsampling target length during training
            if not self.training:
                input_target_len = None
            else:
                if target_len is None:
                    input_target_len = (audio_feat_len / 20).round().long()
                else:
                    input_target_len = target_len
            dsample_results = self.downsampling(inputDict, input_target_len)
        else:
            assert fp_alignment is not None and target_len is not None
            inputDict["fp_alignment"] = fp_alignment
            dsample_results = self.dsample_with_pseudo_weight(inputDict, target_len)
            
        if target_len is not None:
            dsample_results["target_len"] = target_len
            dsample_results["dsample_len_diff"] = (dsample_results["dsample_feats_length"] - target_len).abs().float().mean()
        
        return dsample_results
    
    
    def extract_hidden_states(self, audio_feat, audio_feat_len):
        inputDict = { "audio_feat": audio_feat, "audio_feat_len": audio_feat_len }
        hidden_states, dsample_results = self.forward(inputDict, extract_hiddens=True)
        
        for i in range(audio_feat.shape[0]):
            audio_feat[i] = torch.mul(dsample_results["dsample_alpha"][i].unsqueeze(-1), audio_feat[i])
        hidden_states.append(audio_feat)

        return hidden_states
            
            
    def forward(self, inputDict: dict, extract_hiddens=False):
        audio_feat, audio_feat_len = inputDict["audio_feat"], inputDict["audio_feat_len"]
        audio_feat_pad_mask = get_keypadding_mask(
            max_length=audio_feat.shape[1], data_lens=audio_feat_len
        ).to(audio_feat.device)
        
        if hasattr(self, "self_att"):
            audio_feat, hidden_states = self.self_att.extract_hidden_states(
                src=audio_feat, key_padding_mask=audio_feat_pad_mask
            )
            
        dsample_results = self.downsampling_audio_feat(
            audio_feat=audio_feat,
            audio_feat_len=audio_feat_len,
            audio_feat_pad_mask=audio_feat_pad_mask,
            fp_alignment=inputDict["fp_alignment"] if "fp_alignment" in inputDict else None,
            target_len=inputDict["dsample_target_len"] if "dsample_target_len" in inputDict else None
        )
            
        if extract_hiddens:
            return hidden_states, dsample_results
        
        proj_keywords = self.project_keyword_to_CLIPspace(dsample_results["dsample_feats"])
        cos_score = self.get_keyword_cosine_score(proj_keywords)
        vq_results = self.vector_quantizer(x=cos_score)
        
        assert self.clip.model.token_embedding.weight.requires_grad == False
        vq_keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight
        clip_feats = self.clip.encode_keywords(vq_keywords, dsample_results["dsample_feats_length"])

        result = {
            "cascaded_audio_feat": clip_feats,
            "vq_results": vq_results,
            "vq_keywords": vq_keywords,
            "proj_keywords": proj_keywords,
            "dsample_results": dsample_results,
        }

        return result


class HybridBranch_dynamic(CascadedBranch_dynamic):
    def __init__(
        self, config, audio_dim: int, text_dim: int, out_dim: int, clip: ClipModel
    ) -> None:
        super().__init__(config, audio_dim, text_dim, clip)
        
        self.out_dim = out_dim
        self.parallel_proj = nn.Linear(self.audio_dim, self.out_dim)
        self.cls = self._create_cls()
        logger.info("Start init [CLS] {}".format(self.cls.shape))
        assert hasattr(
            TransformerModels, config.model_settings.parallel_branch.transformer_type
        )
        logger.info(
            f"Using {config.model_settings.parallel_branch.transformer_type} as ParallelBranch"
        )
        self.self_att = getattr(
            TransformerModels, config.model_settings.parallel_branch.transformer_type
        )(**config.model_settings.parallel_branch.transformer_args)

    def _create_cls(self):
        # first cls for parallel objective
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    1,
                    self.config.model_settings.parallel_branch.transformer_args.d_model,
                ]
            )
        )
        
    def forward(self, inputDict: dict, extract_hiddens=False):
        audio_feat, audio_feat_len = inputDict["audio_feat"], inputDict["audio_feat_len"]
        bsz, total_max_len = (
            audio_feat.size(0),
            1 + audio_feat.size(1),
        )
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        audio_feat_pad_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=1 + audio_feat_len
        )

        audio_feat, hidden_states = self.self_att(src=src, key_padding_mask=audio_feat_pad_mask)
        parallel_audio_feat = self.parallel_proj(audio_feat[:, :1].reshape(-1, self.audio_dim))
        audio_feat = audio_feat[:, 1:].reshape(-1, total_max_len - 1, self.audio_dim)
        dsample_results = self.downsampling_audio_feat(
            audio_feat=audio_feat,
            audio_feat_len=audio_feat_len,
            audio_feat_pad_mask=audio_feat_pad_mask,
            fp_alignment=inputDict["fp_alignment"] if "fp_alignment" in inputDict else None,
            target_len=inputDict["dsample_target_len"] if "dsample_target_len" in inputDict else None
        )

        if extract_hiddens:
            hidden_states = [ x[:, 1:, :] for x in hidden_states ]
            return hidden_states, dsample_results
        
        proj_keywords = self.project_keyword_to_CLIPspace(dsample_results["dsample_feats"])
        cos_score = self.get_keyword_cosine_score(proj_keywords)
        vq_results = self.vector_quantizer(x=cos_score)
        
        assert self.clip.model.token_embedding.weight.requires_grad == False
        vq_keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight
        clip_feats = self.clip.encode_keywords(vq_keywords, dsample_results["dsample_feats_length"])

        result = {
            "cascaded_audio_feat": clip_feats,
            "parallel_audio_feat": parallel_audio_feat,
            "vq_keywords": vq_keywords,
            "proj_keywords": proj_keywords,
            "dsample_results": dsample_results,
        }
        
        return result
