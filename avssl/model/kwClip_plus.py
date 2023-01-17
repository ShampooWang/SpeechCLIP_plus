import json
import logging
from turtle import hideturtle

# from audio-visual-ssl.avssl.module.wavlm_modules.WavLM import WavLM

logger = logging.getLogger(__name__)
import math
import os
import pickle
from ast import keyword
from typing import List, Tuple, Union

import numpy as np
import torch
import tqdm
from jiwer import cer, wer
from pytorch_lightning.loggers.wandb import WandbLogger
from s3prl.downstream.specaug import SpecAug
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ..base import OrderedNamespace
from ..data import random_crop_max_length
from ..module import (
    ClipModel,
    Custom_WavLM,
    DiversityLoss,
    FairseqSpeechEncoder_Hubert,
    HybridLoss,
    MLPLayers,
    S3prlSpeechEncoder,
    S3prlSpeechEncoderPlus,
    SimpleCache,
    SupConLoss,
    losses,
    mutualRetrieval,
)
from ..module.fast_vgs_modules import DualEncoder, Wav2Vec2Model_cls
from ..module.kw_modules import TransformerModels
from ..module.kw_modules.TransformerModels import CrossEncoder
from ..module.speechclip_c_modules import vector_quantizers
from ..module.speechclip_c_modules.cif import CIF, CNN, CifMiddleware
from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm, Kw_BatchNorm_plus
from ..optim import get_scheduler
from ..util import freeze_model, get_keypadding_mask
from ..util.embedding_visualization import draw_embedding_space_PCA
from ..util.metric import cosine_semantics
from .base_model import BaseLightningModel
from .kwClip import (
    KW_CascadedBranch,
    KW_CascadedBranch_Integrated,
    KW_ParallelBranch,
    KWClipBase,
)

__all__ = [
    "KWClip_GeneralTransformer_plus",
]

METRIC_REDUCEFN_MAPPING = {
    torch.Tensor: lambda x: torch.mean(x),
    float: lambda x: x,
    int: lambda x: x,
    str: lambda x: x,
}

# ORIG_WAVLM = Custom_WavLM(name="wavlm_base_plus",
#             pretrained=True,
#             trainable=False,
#             feat_select_idx="weighted_sum",
#             layer_drop=0.0,
#             max_audio_len=102400)
# device = torch.device("cuda")
# ORIG_WAVLM = ORIG_WAVLM.to(device)
# ORIG_WAVLM.eval()


class KW_CascadedBranch_plus(nn.Module):
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

        logger.info("Using KW_CascadedBranch_plus")

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
        if self.downsampling_type == "cnn":
            self.downsampling = CNN(
                self.audio_dim, config.model_settings.cascaded_branch.downsampling.cnn
            )
        elif self.downsampling_type == "cif":
            self.using_gt_len = config.model_settings.cascaded_branch.downsampling.get("using_gt_len", False)
            if self.using_gt_len:
                logger.info("Using ground truth text length target")
            self.downsampling = CifMiddleware(
                **config.model_settings.cascaded_branch.downsampling.cif.__dict__
            )
        else:
            raise NotImplementedError(
                "Unknown type:{}".format(config.downsampling.type)
            )
        logger.info("Using {} downsampling method".format(self.downsampling_type))

        if "cross_model" in self.config.model_settings.cascaded_branch.keys():
            logger.info("Using cross model")
            self.cross_encoder = CrossEncoder(self.config.model_settings.cascaded_branch.cross_model)
            if hasattr(self.config.model_settings.cascaded_branch.cross_model, "image_mlp"):
                img_mlp_config = self.config.model_settings.cascaded_branch.cross_model.image_mlp
                logger.info(f"Using cross projection net, dimensions: {img_mlp_config.dimensions}, dropout: {img_mlp_config.dropout}")
                self.cross_proj_net = MLPLayers(
                    units=img_mlp_config.dimensions,
                    dropout=img_mlp_config.dropout,
                )

    def extract_hidden_states(self, audio_feat, audio_len, text=None, use_kw=False):
        if self.downsampling_type == "cnn":
            keywords, new_feat_len = self.downsampling(keywords, audio_len)
        elif self.downsampling_type == "cif":
            if self.using_gt_len and text is not None:
                text_toks_len = []
                for t in text:
                    t = t.squeeze().tolist()
                    _x = t.index(49407)
                    assert _x > 1
                    text_toks_len.append(_x - 1)
            else:
                text_toks_len = [int(_l.item() / 20) for _l in audio_len]

            text_toks_len = torch.tensor(text_toks_len).to(audio_feat.device)
            encoder_outputs = {
                "encoder_raw_out": audio_feat,
                "encoder_padding_mask": get_keypadding_mask(
                    audio_feat.shape[1], audio_len
                ),
            }

            downsampling_out = self.downsampling(encoder_outputs, text_toks_len)
            # return downsampling_out["fired_state"]

            keywords = downsampling_out["cif_out"]
            return keywords
            new_feat_len = downsampling_out["cif_outputs_len"]

        # cif_feat = torch.zeros(audio_feat.shape)
        downsampling_out["cif_weight"] = downsampling_out["cif_weight"].unsqueeze(-1)
        for i in range(audio_feat.shape[0]):
            audio_feat[i] = torch.mul(downsampling_out["cif_weight"][i], audio_feat[i])

        # return downsampling_out["cif_out"]
        return audio_feat


    def forward(self, audio_feat, audio_len, image_feats=None, text=None, return_match_feat=False):
        if self.downsampling_type == "cnn":
            keywords, new_feat_len = self.downsampling(keywords, audio_len)
        elif self.downsampling_type == "cif":
            if self.using_gt_len and text is not None:
                text_toks_len = []
                for t in text:
                    t = t.squeeze().tolist()
                    _x = t.index(49407)
                    assert _x > 1
                    text_toks_len.append(_x - 1)
            else:
                text_toks_len = [int(_l.item() / 20) for _l in audio_len]

            text_toks_len = torch.tensor(text_toks_len).to(audio_feat.device)
            encoder_outputs = {
                "encoder_raw_out": audio_feat,
                "encoder_padding_mask": get_keypadding_mask(audio_feat.shape[1], audio_len),
            }

            downsampling_out = self.downsampling(encoder_outputs, text_toks_len)
            keywords = downsampling_out["cif_out"]
            new_feat_len = downsampling_out["cif_outputs_len"]

        bsz, max_feat_len = keywords.shape[0], keywords.shape[1]

        if hasattr(self, "cross_encoder") and image_feats is not None:
            if hasattr(self, "cross_proj_net"):
                image_feats = self.cross_proj_net(image_feats)
            if return_match_feat:
                max_len, cls_feat_len = keywords.shape[1] + 1, new_feat_len + 1
            else:
                max_len, cls_feat_len = keywords.shape[1], new_feat_len
            keyword_attn_mask = get_keypadding_mask(max_len, cls_feat_len).float()
            keywords, _ = self.cross_encoder(audio_feats=keywords, audio_attention_mask=keyword_attn_mask, vision_feats=image_feats, vision_attention_mask=None, return_cls=return_match_feat)
        if return_match_feat:
            keyword_cls, keywords = keywords[:, 0, :], keywords[:, 1:, :]
            keyword_cls = keyword_cls / keyword_cls.norm(
                dim=-1, keepdim=True
            )
        else:
            keyword_cls = None

        keywords = self.linear_proj(keywords)
        if hasattr(self, "bn_layer"):
            keywords = self.bn_layer(keywords)

        downsampling_out["proj_keywords"] = keywords

        # Cosine
        cos_score = []
        for i in range(max_feat_len):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            max_feat_len,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, max_feat_len, self.clip.model.token_embedding.num_embeddings)}"

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # Feed keyword into clip text encoder
        audio_feat = self.clip.encode_keywords(keywords, new_feat_len)

        result = {
            "cascaded_audio_feat": audio_feat,
            "vq_results": vq_results,
            "vq_keywords": keywords,
            "downsampling_out": downsampling_out,
            "keyword_cls": keyword_cls
        }
        return result


class KW_Hybrid_plus(KW_CascadedBranch_plus):
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
            f"Using {config.model_settings.parallel_branch.transformer_type} as KW_ParallelBranch"
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

    def extract_hidden_states(self, audio_feat, audio_len, use_kw=False):
        bsz, total_max_len = (
            audio_feat.size(0),
            1 + audio_feat.size(1),
        )
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=1 + audio_len
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        if not use_kw:
            hidden_states = [x[:, self.keyword_num + 1 :, ...] for x in hidden_states]
        else:
            hidden_states = [x[:, 1 : self.keyword_num + 1, ...] for x in hidden_states]

        return tuple(hidden_states)

    def forward(self, audio_feat, audio_len, image_feats=None, text=None, return_match_feat=False):
        bsz, total_max_len = (
            audio_feat.size(0),
            1 + audio_feat.size(1),
        )
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=1 + audio_len
        )

        audio_feat = self.self_att(src=src, key_padding_mask=key_padding_mask)
        parallel_cls = audio_feat[:, :1].reshape(-1, self.audio_dim)
        parallel_cls = self.parallel_proj(parallel_cls)
        keywords = audio_feat[:, 1:].reshape(-1, total_max_len - 1, self.audio_dim)
        if self.downsampling_type == "cnn":
            keywords, new_feat_len = self.downsampling(keywords, audio_len)
        elif self.downsampling_type == "cif":
            if self.using_gt_len and text is not None:
                text_toks_len = []
                for t in text:
                    t = t.squeeze().tolist()
                    _x = t.index(49407)
                    assert _x > 1
                    text_toks_len.append(_x - 1)
            else:
                text_toks_len = [int(_l.item() / 20) for _l in audio_len]

            text_toks_len = torch.tensor(text_toks_len).to(audio_feat.device)
            encoder_outputs = {
                "encoder_raw_out": audio_feat,
                "encoder_padding_mask": get_keypadding_mask(audio_feat.shape[1], audio_len),
            }

            downsampling_out = self.downsampling(encoder_outputs, text_toks_len)
            keywords = downsampling_out["cif_out"]
            new_feat_len = downsampling_out["cif_outputs_len"]

        bsz, max_feat_len = keywords.shape[0], keywords.shape[1]

        if hasattr(self, "cross_encoder") and image_feats is not None:
            if hasattr(self, "cross_proj_net"):
                image_feats = self.cross_proj_net(image_feats)
            if return_match_feat:
                max_len, cls_feat_len = keywords.shape[1] + 1, new_feat_len + 1
            else:
                max_len, cls_feat_len = keywords.shape[1], new_feat_len
            keyword_attn_mask = get_keypadding_mask(max_len, cls_feat_len).float()
            keywords, _ = self.cross_encoder(audio_feats=keywords, audio_attention_mask=keyword_attn_mask, vision_feats=image_feats, vision_attention_mask=None, return_cls=return_match_feat)
        if return_match_feat:
            keyword_cls, keywords = keywords[:, 0, :], keywords[:, 1:, :]
            keyword_cls = keyword_cls / keyword_cls.norm(
                dim=-1, keepdim=True
            )
        else:
            keyword_cls = None

        keywords = self.linear_proj(keywords)
        if hasattr(self, "bn_layer"):
            keywords = self.bn_layer(keywords)

        downsampling_out["proj_keywords"] = keywords

        # Cosine
        cos_score = []
        for i in range(max_feat_len):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            max_feat_len,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, max_feat_len, self.clip.model.token_embedding.num_embeddings)}"

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # Feed keyword into clip text encoder
        cascaded_cls = self.clip.encode_keywords(keywords, new_feat_len)

        result = {
            "cascaded_audio_feat": cascaded_cls,
            "parallel_audio_feat": parallel_cls, 
            "vq_results": vq_results,
            "vq_keywords": keywords,
            "downsampling_out": downsampling_out,
            "keyword_cls": keyword_cls
        }
        return result

class KWClipAlignmentBranch(nn.Module):
    def __init__(self, config: OrderedNamespace, audio_dim: int, text_dim: int, clip: ClipModel):
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

        logger.info("Using KW_Alignment_CascadedBranch")

        assert hasattr(
            TransformerModels, config.model_settings.cascaded_branch.transformer_type
        )
        logger.info(
            f"Using {config.model_settings.cascaded_branch.transformer_type} as KW_CascadedBranch"
        )
        self.self_att = getattr(
            TransformerModels, config.model_settings.cascaded_branch.transformer_type
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
        self.layer_norm1 = nn.LayerNorm(audio_dim)
        # self.layer_norm2 = nn.LayerNorm(audio_dim)

    def forward(self, audio_feat, audio_len, alignments, alignment_num):
        # Use multi-head attention layer to find keywords(cls)
        bsz, edim = audio_feat.shape[0], audio_feat.shape[-1]
        # audio_feat = self.layer_norm1(audio_feat)
        max_feat_len = torch.max(alignment_num)
        # print(alignment_num)
        src = torch.zeros((bsz, max_feat_len, edim)).to(audio_feat.device)
        for i in range(bsz):
            _ali_num = alignment_num[i]
            for j, _seg in enumerate(alignments[i]):
                _ts, _te = _seg[0], _seg[1]
                if _ts == -1:
                    break

                if _te == audio_len[i]:
                    src[i, j, :] = torch.mean(audio_feat[i, _ts:, :], dim=0)
                else: 
                    src[i, j, :] = torch.mean(audio_feat[i, _ts:_te, :], dim=0)

        key_padding_mask = get_keypadding_mask(
            max_length=max_feat_len, data_lens=alignment_num
        )

        src = self.layer_norm1(src)
        # print(src)
        keywords = self.self_att(src=src, key_padding_mask=key_padding_mask)
        keywords = self.linear_proj(keywords)

        if hasattr(self, "bn_layer"):
            keywords = self.bn_layer(keywords)

        # Cosine
        cos_score = []
        for i in range(max_feat_len):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(bsz, self.text_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )

        cos_score = torch.stack(cos_score, dim=1)

        assert cos_score.shape == (
            bsz,
            max_feat_len,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, max_feat_len, self.clip.model.token_embedding.num_embeddings)}"

        # VQ
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        # Feed keyword into clip text encoder
        audio_feat = self.clip.encode_keywords(keywords, alignment_num)

        result = {
            "cascaded_audio_feat": audio_feat,
            "vq_results": vq_results,
            "vq_keywords": keywords,
            "keywords_len": alignment_num,
        }
        return result 


class KWClip_GeneralTransformer_plus(KWClipBase):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        self.cascaded_branch = None
        self.parallel_branch = None

        if self.config.model_settings.cascaded_objective_weight > 0:
            logger.info("Create Cascaded plus Branch")
            if (
                self.config.model_settings.cascaded_branch.type
                == "KW_CascadedBranch_plus"
            ):
                self.cascaded_branch = KW_CascadedBranch_plus(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif self.config.model_settings.cascaded_branch.type == "KW_Hybrid_plus":
                assert self.config.model_settings.parallel_objective_weight > 0
                logger.info("Using Parallel Objective (Integrated w/ cascaded_branch)")
                self.cascaded_branch = KW_Hybrid_plus(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    out_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif self.config.model_settings.cascaded_branch.type == "KWClipAlignmentBranch":
                logger.info("Using Ground truth alignment")
                self.cascaded_branch = KWClipAlignmentBranch(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            else:
                raise NotImplementedError(
                    self.config.model_settings.cascaded_branch.type
                )
            if hasattr(self.cascaded_branch, "downsampling") and self.cascaded_branch.downsampling.cal_quantity_loss:
                self.quantity_loss_criteria = nn.L1Loss()

        if (
            self.config.model_settings.parallel_objective_weight > 0
            and not self.config.model_settings.cascaded_branch.type == "KW_Hybrid_plus"
        ):
            logger.info("Create Parallel plus Branch")
            self.parallel_branch = KW_ParallelBranch(
                config=self.config,
                audio_dim=self.audio_embd_dim,
                out_dim=self.subword_embd_dim,
            )

        self.img_enc_proj_net = None
        image_encoder_projection = self.config.model_settings.get(
            "image_encoder_projection", None
        )
        if image_encoder_projection is not None:
            logger.info(
                f"image_encoder_projection dims:{image_encoder_projection.dimensions} droupout:{image_encoder_projection.dropout}"
            )
            self.img_enc_proj_net = MLPLayers(
                units=image_encoder_projection.dimensions,
                dropout=image_encoder_projection.dropout,
            )

        self.p_branch_proj_net = None
        parallel_branch_projection = self.config.model_settings.get(
            "parallel_branch_projection", None
        )
        if parallel_branch_projection is not None:
            logger.info(
                f"parallel_branch_projection dims:{parallel_branch_projection.dimensions} droupout:{parallel_branch_projection.dropout}"
            )
            self.p_branch_proj_net = MLPLayers(
                units=parallel_branch_projection.dimensions,
                dropout=parallel_branch_projection.dropout,
            )

        self.c_branch_proj_net = None
        cascaded_branch_projection = self.config.model_settings.get(
            "cascaded_branch_projection", None
        )
        if cascaded_branch_projection is not None:
            logger.info(
                f"cascaded_branch_projection dims:{cascaded_branch_projection.dimensions} droupout:{cascaded_branch_projection.dropout}"
            )
            self.c_branch_proj_net = MLPLayers(
                units=cascaded_branch_projection.dimensions,
                dropout=cascaded_branch_projection.dropout,
            )

        if self.config.audio_encoder.get("regularization", False):
            from ..module.output_regularization import Audio_encoder_regularization
            logger.info("Using audio encoder regularization")
            self.ae_reg_criterion = Audio_encoder_regularization(config.audio_encoder)

        self.keyword_objective_weight = config.model_settings.get("keyword_objective_weight", 0)

        self.matching_objective_weight = config.model_settings.get("matching_objective_weight", 0)
        if self.matching_objective_weight > 0:
            logger.info("Adding matching objective")
            assert hasattr(config.model_settings, "matching_mlp")
            mlp_cfg = config.model_settings.matching_mlp
            assert mlp_cfg.dimensions[-1] == 1
            assert len(mlp_cfg.dimensions) == 3
            self.matching_mlp = MLPLayers(units=mlp_cfg.dimensions, nonlin=nn.GELU(), dropout=mlp_cfg.dropout)
            self.matching_loss = nn.CrossEntropyLoss()

        self.keyword_diversity_weight = config.model_settings.cascaded_branch.keyword.get("diversity_weight", 0)
        if self.keyword_diversity_weight > 0: 
            logger.info("Adding keyword diversity objective")
            self.keyword_diversity_type = config.model_settings.cascaded_branch.keyword.get("diversity_type", "ent")
            logger.info(f"Keyword diversity type: {self.keyword_diversity_type}")
            if self.keyword_diversity_type == "corr" or self.keyword_diversity_type == "cos":
                self.keyword_diversity_criterion = DiversityLoss(self.keyword_diversity_weight)
            elif self.keyword_diversity_type == "ent":
                pass
            else:
                raise NotImplementedError(self.keyword_diversity_type)

    def getTrainableParams(self):
        _params = super().getTrainableParams()
        if self.cascaded_branch is not None:
            logger.info("Add cascaded_plus_branch parameters")
            _params += list(self.cascaded_branch.parameters())

        if self.parallel_branch is not None:
            logger.info("Add parallel_branch parameters")
            _params += list(self.parallel_branch.parameters())

        if self.img_enc_proj_net is not None:
            logger.info("Add img_enc_proj_net parameters")
            _params += list(self.img_enc_proj_net.parameters())

        if self.p_branch_proj_net is not None:
            logger.info("Add parallel_branch_projection parameters")
            _params += list(self.p_branch_proj_net.parameters())

        if hasattr(self, "matching_mlp"):
            _params += list(self.matching_mlp.parameters())

        return _params

    def forward_image(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(images, list):
            image_tensor = self.clip.prep_image(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            if images.dim() != 4 or images.shape[1] != 3:
                raise ValueError(f"Incorrect image tensor shape {images.shape}")
            image_tensor = images
        else:
            raise TypeError(f"Unknown image type {type(images)}")

        if hasattr(self.cascaded_branch, "cross_encoder"):
            return self.clip.extract_image_features(image_tensor)

        image_feat = self.clip.encode_image(image_tensor)
        return image_feat, None

    def cal_matching_loss(self, feat1, feat2):
        bsz = feat1.shape[0]
        labels = torch.ones(bsz, dtype=torch.long).to(feat1.device) # 0: mismatch, 1: match
        mismatch_idx = np.random.choice(bsz, int(bsz*0.5), replace=False)
        labels[mismatch_idx] = 0
        match_idx = np.setdiff1d(np.arange(bsz), mismatch_idx)
        # feat1_clone = feat1.clone()
        feat1[mismatch_idx] = feat1[match_idx[:int(bsz*0.5)]]
        relation_feat = torch.stack([feat1, feat2], dim=1)
        match_logits = self.matching_mlp(relation_feat).squeeze(-1)
        matching_loss = self.matching_loss(match_logits, labels)
        matching_acc = torch.sum(torch.argmax(match_logits, dim=1) == labels) / bsz

        return matching_loss, matching_acc

    def forward(
        self,
        batch,
        cal_loss: bool = False,
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]

        # update device information to clip model
        self.clip.update_device(self.device)

        image_feat, image_features  = self.forward_image(image)
        if self.img_enc_proj_net is not None:
            image_feat = self.img_enc_proj_net(image_feat)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        audio_feat, audio_len, hidden_states = self.forward_audio(
            wav, wav_len, return_hidden_states=True
        )

        if hasattr(self, "spec_aug"):
            audio_feat = [x for x in audio_feat]
            audio_feat, _ = self.spec_aug(audio_feat)
            audio_feat = torch.stack(audio_feat, 0)

        if self.cascaded_branch is not None:
            if isinstance(self.cascaded_branch, KWClipAlignmentBranch):
                cascaded_result = self.cascaded_branch(
                    audio_feat=audio_feat,
                    audio_len=audio_len,
                    alignments=batch["alignments"],
                    alignment_num=batch["alignment_num"]
                )
            else:
                if self.cascaded_branch.using_gt_len:
                    text = batch["text"]
                else:
                    text = None
                cascaded_result = self.cascaded_branch(
                    audio_feat=audio_feat,
                    audio_len=audio_len,
                    image_feats=image_features,
                    text=text,
                    return_match_feat=hasattr(self, "matching_mlp")
                )
            cascaded_audio_feat = cascaded_result.get("cascaded_audio_feat", None)
            parallel_audio_feat = cascaded_result.get("parallel_audio_feat", None)
            vq_results = cascaded_result.get("vq_results", None)
            vq_keywords = cascaded_result.get("vq_keywords", None)
            downsampling_out = cascaded_result.get("downsampling_out", None)
            keyword_cls = cascaded_result.get("keyword_cls", None)

            if downsampling_out is None:
                assert "keywords_len" in cascaded_result.keys(), cascaded_result.keys()
                keywords_len = cascaded_result["keywords_len"]

        if self.parallel_branch is not None:
            parallel_result = self.parallel_branch(
                audio_feat=audio_feat, audio_len=audio_len, img_feat=image_feat
            )
            parallel_audio_feat = parallel_result.get("parallel_audio_feat", None)
            if self.p_branch_proj_net is not None:
                parallel_audio_feat = self.p_branch_proj_net(parallel_audio_feat)

        losses = {
            "id": id,
            "image_feat": image_feat,
        }

        if hasattr(self, "ae_reg_criterion"):
            losses["ae_reg_loss"] = self.ae_reg_criterion(wav, wav_len, audio_feat)

        if cascaded_audio_feat is not None:
            cascaded_audio_feat = cascaded_audio_feat / cascaded_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["cascaded_audio_feat"] = cascaded_audio_feat

        if parallel_audio_feat is not None:
            parallel_audio_feat = parallel_audio_feat / parallel_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["parallel_audio_feat"] = parallel_audio_feat

        # update log_metrics
        log_metrics = {}
        log_metrics.update(
            {
                "cl_temp": self.criterion.current_temperature,
            }
        )

        if downsampling_out is not None:
            proj_keywords = downsampling_out.get("proj_keywords", None)
            keywords_len = downsampling_out.get("cif_outputs_len", None)
            quantity_out = downsampling_out.get("quantity_out", None)
            target_len = downsampling_out.get("target_len", None)
            cif_output_len_diff = downsampling_out.get("cif_output_len_diff", None)

            if quantity_out is not None and target_len is not None:
                losses["cif_quantity_out"] = quantity_out
                losses["cif_target_len"] = target_len

            if self.keyword_diversity_weight > 0:
                if self.keyword_diversity_type == "ent":
                    assert "diversity_loss" in vq_results, "entropy loss is not in vq_results"
                    losses["keyword_diversity_loss"] = vq_results["diversity_loss"]
                elif self.keyword_diversity_type == "corr":
                    if (
                        vq_keywords is not None
                        and keywords_len is not None
                        and hasattr(self, "keyword_diversity_criterion")
                    ):
                        losses["keyword_diversity_loss"] = self.keyword_diversity_criterion(
                            vq_keywords, keywords_len, self.keyword_diversity_type
                        )
                elif self.keyword_diversity_type == "cos":
                    if (
                        vq_keywords is not None
                        and keywords_len is not None
                        and hasattr(self, "keyword_diversity_criterion")
                    ):
                        losses["keyword_diversity_loss"] = self.keyword_diversity_criterion(
                            vq_keywords, keywords_len, self.keyword_diversity_type
                        )

            if self.keyword_objective_weight > 0 and proj_keywords is not None:
                keywords_audio_feat = torch.mean(proj_keywords, dim=1)
                assert keywords_audio_feat.shape == image_feat.shape, f"kw_feat: {keywords_audio_feat.shape}, img_feat: {image_feat.shape}"
                keywords_audio_feat = keywords_audio_feat / keywords_audio_feat.norm(
                    dim=-1, keepdim=True
                )
                losses["keywords_audio_feat"] = keywords_audio_feat

            if self.matching_objective_weight > 0:
                matching_loss, matching_acc = self.cal_matching_loss(keyword_cls, cascaded_audio_feat)
                losses["matching_loss"] = matching_loss
                log_metrics["matching_acc"] = matching_acc
            
            if cif_output_len_diff is not None:
                log_metrics["cif_output_len_diff"] = sum(cif_output_len_diff) / len(cif_output_len_diff)

        if self.config.model_settings.cascaded_objective_weight > 0:
            log_metrics["softmax_temp"] = vq_results["temp"]
            log_metrics["code_perplexity"] = vq_results["code_perplexity"]
            log_metrics["prob_perplexity"] = vq_results["prob_perplexity"]
            log_metrics["ent_per_t"] = vq_results["ent_per_t"]

        return (
            losses,
            log_metrics,
            {
                "cascaded_audio_feat": cascaded_audio_feat,
                "parallel_audio_feat": parallel_audio_feat,
                "image_feat": image_feat,
                "id": id,
                "vq_results": vq_results,
                "vq_keywords": vq_keywords,
                "keywords_len": keywords_len,
            },
        )

    def compute_loss(self, input_feats):
        """compute the loss here

        Args:
            input_feats (Any): the feats required for computing loss
        """
        assert isinstance(input_feats, dict)
        assert "id" in input_feats
        assert (
            "cascaded_audio_feat" in input_feats or "parallel_audio_feat" in input_feats
        )
        assert "image_feat" in input_feats

        cascaded_audio_feat = (
            input_feats["cascaded_audio_feat"].float()
            if "cascaded_audio_feat" in input_feats
            else None
        )
        parallel_audio_feat = (
            input_feats["parallel_audio_feat"].float()
            if "parallel_audio_feat" in input_feats
            else None
        )
        keywords_audio_feat = (
            input_feats["keywords_audio_feat"].float()
            if "keywords_audio_feat" in input_feats
            else None
        )
        image_feat = input_feats["image_feat"].float()
        id = input_feats["id"]
        ae_reg_loss = (
            input_feats["ae_reg_loss"].float() if "ae_reg_loss" in input_feats else None
        )
        keyword_diversity_loss = (
            input_feats["keyword_diversity_loss"].float()
            if "keyword_diversity_loss" in input_feats
            else None
        )
        matching_loss = (
            input_feats["matching_loss"].float()
            if "matching_loss" in input_feats
            else None
        )

        losses = {"loss": 0}
        if self.config.model_settings.cascaded_objective_weight > 0:
            losses["c_cl_loss"] = self.criterion(
                feat_A=cascaded_audio_feat,
                feat_B=image_feat,
                index=id,
            )
            losses["loss"] += (
                self.config.model_settings.cascaded_objective_weight
                * losses["c_cl_loss"]
            )

        if self.config.model_settings.parallel_objective_weight > 0:
            losses["p_cl_loss"] = self.criterion(
                feat_A=parallel_audio_feat,
                feat_B=image_feat,
                index=id,
            )
            losses["loss"] += (
                self.config.model_settings.parallel_objective_weight
                * losses["p_cl_loss"]
            )
        
        if self.config.model_settings.keyword_objective_weight > 0:
            losses["k_cl_loss"] = self.criterion(
                feat_A=keywords_audio_feat,
                feat_B=image_feat,
                index=id,
            )
            losses["loss"] += (
                self.config.model_settings.keyword_objective_weight
                * losses["k_cl_loss"]
            )

        if "cif_quantity_out" in input_feats and "cif_target_len" in input_feats and hasattr(self, "quantity_loss_criteria"):
            quantity_out = input_feats["cif_quantity_out"]
            target_len = input_feats["cif_target_len"]
            losses["quantity_loss"] = self.quantity_loss_criteria(
                quantity_out, target_len
            )
            losses["loss"] += losses["quantity_loss"]

        if hasattr(self, "ae_reg_criterion"):
            losses["ae_reg_loss"] = torch.mean(ae_reg_loss)
            losses["loss"] += self.ae_reg_criterion.weight * losses["ae_reg_loss"]

        if self.keyword_diversity_weight > 0 and keyword_diversity_loss is not None:
            losses["keyword_diversity_loss"] = torch.mean(keyword_diversity_loss)
            losses["loss"] += (
                self.keyword_diversity_weight
                * losses["keyword_diversity_loss"]
            )
        if self.matching_objective_weight > 0 and matching_loss is not None:
            losses["matching_loss"] = torch.mean(matching_loss)
            losses["loss"] += (
                self.matching_objective_weight
                * losses["matching_loss"]
            )
        return losses

    def validation_step(self, batch, batch_idx):
        losses, log_metrics, others = self.forward(batch)

        audio_feat = (
            others["cascaded_audio_feat"]
            if self.config.retrieval.audio_feat_src == "cascaded"
            else others["parallel_audio_feat"]
        )

        image_feat = others["image_feat"] if "image_feat" in others else None
        text_feat = others["text_feat"] if "text_feat" in others else None
        id = others["id"]

        return_dict = {
            "id": id,
            "audio_feat": audio_feat,
        }

        if image_feat is not None:
            return_dict["image_feat"] = image_feat
        if text_feat is not None:
            return_dict["text_feat"] = text_feat

        if "vq_keywords" in others and others["vq_keywords"] is not None:
            vq_keywords = others["vq_keywords"]
            bsz, max_feat_len, embd_dim = vq_keywords.size()
            return_dict["vq_keywords"] = vq_keywords.view(-1, embd_dim)
            return_dict["keywords_bsz"] = bsz
            return_dict["max_feat_len"] = max_feat_len
            return_dict["gold_text"] = batch["text"]

        if "keywords_len" in others and others["keywords_len"] is not None:
            return_dict["keywords_len"] = others["keywords_len"]
        # if "vq_results" in others and others["vq_results"]["targets"] is not None:
        #     return_dict["targets"] = others["vq_results"]["targets"].view(-1, embd_dim)

        return {"loss_feats": losses, "log_metrics": log_metrics, "others": return_dict}

    def validation_step_end(self, outputs):
        assert isinstance(outputs, dict)
        losses = self.compute_loss(outputs["loss_feats"])

        log_metrics = outputs["log_metrics"]
        result = {
            **{f"val_{k}": losses[k] for k in losses},
            **{
                f"val_{k}": METRIC_REDUCEFN_MAPPING[type(log_metrics[k])](
                    log_metrics[k]
                )
                for k in log_metrics
            },
        }
        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for k in outputs["others"]:
            if isinstance(outputs["others"][k], torch.Tensor):
                outputs["others"][k] = outputs["others"][k].detach().cpu()
        return outputs["others"]

    def validation_epoch_end(self, outputs):
        if "vq_keywords" in outputs[0].keys():
            if not os.path.exists(
                os.path.join(self.config.trainer.default_root_dir, "retokenizeText")
            ):
                os.makedirs(
                    os.path.join(
                        self.config.trainer.default_root_dir, "retokenizeText"
                    ),
                    exist_ok=True,
                )
            if not os.path.exists(
                os.path.join(self.config.trainer.default_root_dir, "visualization")
            ):
                os.makedirs(
                    os.path.join(self.config.trainer.default_root_dir, "visualization"),
                    exist_ok=True,
                )

            if (
                hasattr(self, "log_detokenize_results_every_n_epoch")
                and self.current_epoch % self.log_detokenize_results_every_n_epoch == 0
            ) or not (hasattr(self, "log_detokenize_results_every_n_epoch")):
                gold_texts, keyword_embeddings_list, feat_len_list = [], [], []
                for x in outputs:
                    for sent, _l in zip(x["gold_text"], x["keywords_len"]):
                        # exit(1)
                        gold_texts.append(
                            self.clip.tokenizer.decode(sent.squeeze().tolist())
                        )
                        feat_len_list.append(_l.item())

                    keyword_bsz = x["keywords_bsz"].tolist()
                    keyword_num = x["max_feat_len"].tolist()
                    assert len(keyword_bsz) == len(keyword_num)

                    start = 0
                    tmp_embd_list = []
                    for _bsz, _knum in zip(keyword_bsz, keyword_num):
                        tmp_embd_list.append(
                            x["vq_keywords"][start : start + _bsz * _knum].view(
                                _bsz, _knum, -1
                            )
                        )
                        start += _bsz * _knum

                    keyword_embeddings_list.append(tmp_embd_list)
                    # gold_texts.extend(self.clip.deTokenize(x["gold_text"]))
                # gold_texts = [ x["gold_text"] for x in outputs]
                # gold_texts = [ x["gold_text"] for x in gold_texts]
                assert len(gold_texts) == len(
                    feat_len_list
                ), f"gold_texts {len(gold_texts)}, feat_len_list {len(feat_len_list)}"
                all_keyword_embeddings = torch.cat(
                    [x["vq_keywords"] for x in outputs], dim=0
                )

                # all_keyword_embeddings shape (total_audio, num_keywords, hid_dim)
                embeddings_stat_dict = {
                    "mean": {},
                    "std": {},
                    "norm": {},
                }
                tokenEmbeddings = self.clip.model.token_embedding.weight.detach().cpu()

                # calculate mean, variance
                embeddings_stat_dict["mean"][f"kw"] = torch.mean(
                    torch.mean(all_keyword_embeddings, dim=0)
                )
                embeddings_stat_dict["std"][f"kw"] = torch.mean(
                    torch.std(all_keyword_embeddings, dim=0)
                )
                embeddings_stat_dict["norm"][f"kw"] = torch.mean(
                    torch.norm(all_keyword_embeddings, p=2, dim=-1)
                )

                embeddings_stat_dict["mean"]["pretrained"] = torch.mean(
                    torch.mean(tokenEmbeddings, dim=0)
                )
                embeddings_stat_dict["std"]["pretrained"] = torch.mean(
                    torch.std(tokenEmbeddings, dim=0)
                )
                embeddings_stat_dict["norm"]["pretrained"] = torch.mean(
                    torch.norm(tokenEmbeddings, p=2, dim=-1)
                )

                # self.log("embs_mean", embeddings_stat_dict["mean"])
                # self.log("embs_std", embeddings_stat_dict["std"])
                # self.log("embs_norm", embeddings_stat_dict["norm"])

                self.log(
                    "kw_mean_mse",
                    torch.norm(
                        torch.mean(
                            all_keyword_embeddings.view(-1, self.subword_embd_dim),
                            dim=0,
                        )
                        - torch.mean(tokenEmbeddings, dim=0),
                        p=2,
                    ),
                    sync_dist=True,
                )
                # self.log("kw_std_mse",torch.std(
                #     torch.norm(
                #         torch.std(all_keyword_embeddings.view(-1,self.subword_embd_dim),dim=0) - torch.std(tokenEmbeddings,dim=0),p=2
                #     )
                # ))

                draw_embedding_space_PCA(
                    kw_embs=all_keyword_embeddings,
                    gold_embs=tokenEmbeddings,
                    output_path=os.path.join(
                        self.config.trainer.default_root_dir,
                        "visualization/",
                        "pca_ep{}.pdf".format(self.current_epoch),
                    ),
                )

                if not hasattr(self.config.log_setting, "log_draw_pca_every_n_epoch"):
                    self.config.log_setting.log_draw_pca_every_n_epoch = 0

                if self.config.log_setting.log_draw_pca_every_n_epoch > 0:
                    if (
                        self.current_epoch
                        % self.config.log_setting.log_draw_pca_every_n_epoch
                        == 0
                    ):
                        draw_embedding_space_PCA(
                            kw_embs=all_keyword_embeddings,
                            gold_embs=tokenEmbeddings,
                            output_path=os.path.join(
                                self.config.trainer.default_root_dir,
                                "visualization/",
                                "pca_ep{}.pdf".format(self.current_epoch),
                            ),
                        )

                assert all_keyword_embeddings.dim() == 2, all_keyword_embeddings.shape
                assert (
                    all_keyword_embeddings.shape[-1] == self.subword_embd_dim
                ), all_keyword_embeddings.shape

                all_retok_outputs = []

                K = self.config.model_settings.cascaded_branch.keyword.get(
                    "detokenized_K_neighbors", 10
                )

                if not hasattr(
                    self.config.model_settings.cascaded_branch.keyword,
                    "retrieve_method",
                ):
                    self.config.model_settings.cascaded_branch.keyword.retrieve_method = (
                        "cosine"
                    )

                if (
                    self.config.model_settings.cascaded_branch.keyword.retrieve_method
                    == "pseudo_inverse"
                ):
                    emb_pinv = torch.linalg.pinv(tokenEmbeddings.T).float()

                assert (
                    self.config.model_settings.cascaded_branch.keyword.retrieve_method
                    in ["cosine", "pseudo_inverse"]
                )

                # emb_pinv.shape (num of codes, dim)
                # kw_top_ret = [[] for _ in range(self.keyword_num)]
                print("Detokenizing K={}".format((K)))

                hit_rate_list = []
                for b_idx, i in zip(
                    range(len(outputs)),
                    range(0, len(gold_texts), self.config.data.dev_batch_size),
                ):
                    _gold_texts = gold_texts[i : i + self.config.data.dev_batch_size]
                    _feat_len_list = feat_len_list[
                        i : i + self.config.data.dev_batch_size
                    ]
                    gold_subword_toks_set = [
                        set(self.clip.tokenizer.encode(_text)) for _text in _gold_texts
                    ]
                    # if len(_gold_texts) == 0:
                    #     break

                    b_instance_idx = 0
                    for keyword_embeddings in keyword_embeddings_list[b_idx]:
                        _bsz, _max_feat_len = (
                            keyword_embeddings.shape[0],
                            keyword_embeddings.shape[1],
                        )
                        # _bsz = len(_gold_texts)
                        _k_values, _k_indices = torch.topk(
                            (
                                emb_pinv.float()
                                @ keyword_embeddings.view(-1, self.subword_embd_dim)
                                .float()
                                .reshape(-1, self.subword_embd_dim)
                                .permute(1, 0)
                            ).permute(1, 0)
                            if self.config.model_settings.cascaded_branch.keyword.retrieve_method
                            == "pseudo_inverse"
                            else F.cosine_similarity(
                                keyword_embeddings.view(-1, self.subword_embd_dim, 1),
                                tokenEmbeddings.transpose(0, 1).unsqueeze(0),
                                dim=1,
                            ),
                            K,
                        )

                        assert _k_values.shape == (
                            _bsz * _max_feat_len,
                            K,
                        ), _k_values.shape
                        _k_indices = _k_indices.view(_bsz, _max_feat_len, K)
                        _k_values = _k_values.view(_bsz, _max_feat_len, K)

                        for x in range(_bsz):
                            _hit_rate = 0
                            hit_kw = []
                            tmp_outputs = {}
                            _feat_len = _feat_len_list[b_instance_idx + x]
                            _gold_subword_toks_set = gold_subword_toks_set[
                                b_instance_idx + x
                            ]
                            for _keyword_i in range(_feat_len):
                                tmp_outputs["keyword_{}".format(_keyword_i)] = []

                                # check if nearest K subword appears in gold text
                                top_k_toks = set(
                                    [
                                        self.clip.reducedl2Original[_ind.item()]
                                        if self.clip.selected_text_emb_ids is not None
                                        else _ind.item()
                                        for _ind in _k_indices[x, _keyword_i]
                                    ]
                                )

                                if bool(top_k_toks & _gold_subword_toks_set):
                                    _hit_rate += 1 / _feat_len
                                    hit_token_id = int(
                                        list(top_k_toks & _gold_subword_toks_set)[0]
                                    )
                                    hit_token = self.clip.tokenizer.decoder[
                                        hit_token_id
                                        if self.clip.selected_text_emb_ids is not None
                                        else self.clip.reducedl2Original[hit_token_id]
                                    ]
                                    hit_kw.append(hit_token)

                                for _ind, _dist in zip(
                                    _k_indices[x, _keyword_i], _k_values[x, _keyword_i]
                                ):
                                    tmp_outputs["keyword_{}".format(_keyword_i)].append(
                                        [
                                            self.clip.tokenizer.decoder[
                                                self.clip.reducedl2Original[_ind.item()]
                                                if self.clip.selected_text_emb_ids
                                                is not None
                                                else _ind.item()
                                            ],
                                            _dist.item(),
                                        ]
                                    )
                            # print(_hit_rate)
                            hit_rate_list.append(_hit_rate)

                            all_retok_outputs.append(
                                {
                                    "gold": _gold_texts[b_instance_idx + x],
                                    "neighbors": tmp_outputs,
                                    "hit_kw": hit_kw,
                                    "kw_hit_rate": _hit_rate,
                                }
                            )
                        b_instance_idx += _bsz

                val_kw_hit_rate = sum(hit_rate_list) / len(hit_rate_list) * 100
                print("val_kw_hit_rate", val_kw_hit_rate)
                self.log(
                    "val_kw_hit_rate",
                    val_kw_hit_rate,
                    sync_dist=True,
                )

                cos_semantic_list = cosine_semantics(all_retok_outputs)
                val_cos_semantics = sum(cos_semantic_list) / len(cos_semantic_list)
                print("val_cos_semantics", val_cos_semantics)
                self.log(
                    "val_cos_semantics",
                    val_cos_semantics,
                    sync_dist=True,
                )

                with open(
                    os.path.join(
                        self.config.trainer.default_root_dir,
                        "retokenizeText/",
                        "keywords_ep{}.json".format(self.current_epoch),
                    ),
                    "w",
                ) as f:
                    json.dump(all_retok_outputs, f)
                del all_retok_outputs

        all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        all_imgs = torch.cat([x["image_feat"] for x in outputs], dim=0)
        id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_imgs)}

        del all_imgs

        all_audo_feats = torch.cat([x["audio_feat"] for x in outputs], dim=0)
        all_audo_feats_id = all_ids

        all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
        all_img_feats_id = torch.LongTensor(list(id_img_pairs.keys()))

        torch.save(
            all_audo_feats.detach().cpu(),
            os.path.join(self.config.trainer.default_root_dir, "all_audio_feats.pt"),
        )
        torch.save(
            all_img_feats.detach().cpu(),
            os.path.join(self.config.trainer.default_root_dir, "all_img_feats.pt"),
        )

        print(
            "Total #{} images, #{} audio".format(
                len(all_img_feats), len(all_audo_feats)
            )
        )

        # calculate dot product
        score_per_audio = torch.matmul(
            all_audo_feats.float().to(self.device),
            all_img_feats.float().T.to(self.device),
        )
        score_per_audio = score_per_audio / 0.07
        score_per_image = score_per_audio.T

        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_img_feats_id

        self.reportRetrieval(
            score_per_audio=score_per_audio,
            score_per_image=score_per_image,
            AI_answers=AI_answers,
            IA_answers=IA_answers,
        )

    def feature_extractor_s3prl(self, wav, featrure_layer_norm=True, add_cif=False):
        wav, wav_len = self.processWavs(wav)

        # _, _, wavlm_hidden_states = ORIG_WAVLM(
        #         wav, wav_len, return_hidden_states=True
        # )
        # assert isinstance(wavlm_hidden_states, tuple)

        audio_feat, audio_len, hidden_states = self.forward_audio(
            wav, wav_len, return_hidden_states=True
        )
        assert isinstance(hidden_states, tuple)

        # assert hidden_states[0].shape == wavlm_hidden_states[0].shape, f"c+: {hidden_states[0].shape}, wavlm: {wavlm_hidden_states[0].shape}"
        seq_len = hidden_states[0].shape[1]
        # q = seq_len // self.keyword_num
        # r = seq_len % self.keyword_num
        # repeats = (self.keyword_num - 1) * [q] + [ q+r ]
        # repeats = torch.tensor(repeats, device=hidden_states[0].device)

        cascaded_hidden_states = None
        parallel_hidden_states = None
        if add_cif:
            if self.cascaded_branch is not None:
                cascaded_hidden_states = self.cascaded_branch.extract_hidden_states(
                    audio_feat, audio_len, use_kw=False
                )
                hidden_states = [torch.cat([x, cascaded_hidden_states], dim=1) for x in hidden_states]
            # wavlm_hidden_states = [x for x in wavlm_hidden_states]
            # if cascaded_hidden_states.dim() == 2:
            #     cascaded_hidden_states = cascaded_hidden_states.unsqueeze(1)
            #     cascaded_hidden_states = cascaded_hidden_states.repeat(1, seq_len, 1)
            # else:
            # cascaded_hidden_states = torch.mean(cascaded_hidden_states, dim=1, keepdim=True).repeat(1, seq_len, 1)

            # assert isinstance(cascaded_hidden_states, tuple)
            # dup_kw = (torch.repeat_interleave(x, repeats, dim=1) for x in cascaded_hidden_states[1:])
            # hidden_states = cascaded_hidden_states
            # hidden_states = hidden_states + wavlm_hidden_states
            # hidden_states = [torch.cat((x, y), dim=-1) for x, y in zip(wavlm_hidden_states, hidden_states)]

        if self.parallel_branch is not None:
            parallel_hidden_states = self.parallel_branch.extract_hidden_states(
                audio_feat, audio_len, use_kw=False
            )
            assert isinstance(parallel_hidden_states, tuple)
            # dup_kw = (torch.repeat_interleave(x, repeats, dim=1) for x in cascaded_hidden_states[1:])
            # hidden_states = hidden_states + tuple(dup_kw)
            hidden_states = hidden_states + tuple(parallel_hidden_states[1:-1])

        # assert len(hidden_states) == 15
        # print(hidden_states[0].shape)
        # print(hidden_states[-1].shape)
        # if hidden_states[0].shape[0] > 1:
        # assert hidden_states[0].shape[0] == 1
        # import uuid
        # import glob

        # current_files_num = len(list(glob.glob("/work/twsezjg982/atosystem/audio-visual-ssl/slurms/KS_hidstates/KW_bsz256_WS_p1_flickr/*.pt")))
        # if current_files_num >= 51094:
        #     print("Finish")
        #     exit(1)

        # hubert_states = torch.stack(hidden_states).view(14,-1,768)
        # hubert_states = torch.mean(torch.norm(hubert_states,dim=-1),dim=-1)
        # assert hubert_states.shape == (14,)
        # # gap = torch.mean(torch.norm(hubert_states[:-1,...] - hubert_states[-1,...],dim=-1),dim=-1)
        # # print(hubert_states.shape)
        # # exit(1)
        # torch.save(hubert_states.cpu(),f"/work/twsezjg982/atosystem/audio-visual-ssl/slurms/KS_hidstates/KW_bsz256_WS_p1_flickr/{uuid.uuid4()}.pt")

        # hidden_states = [torch.mean(x, dim=1, keepdim=True).repeat(1, seq_len, 1) for x in hidden_states]
        assert featrure_layer_norm == True
        if featrure_layer_norm:
            hidden_states = torch.stack(hidden_states, dim=0)
            hidden_states = F.layer_norm(hidden_states, (hidden_states.shape[-1],))

        # If only one feature for hiddenstates
        # hidden_states = [hidden_states]
        # else:
        hidden_states = [x for x in hidden_states]

        return hidden_states[-1], hidden_states

    def feature_extractor_zerospeech(self, wav):
        result = []
        embeddings, feat_len = self.forward_audio(wav)
        if self.audio_encoder.feat_select_idx != "cif":
            for _embs, _len in zip(embeddings, feat_len):
                result.append(_embs[:_len].cpu().float().numpy())
        else:
            keywords = self.cascaded_branch.extract_hidden_states(embeddings, feat_len)
            for _k in keywords:
                assert _k.dim() == 2
                if _k.shape[0] == 1:
                    _k = _k.repeat(2, 1)
                result.append(_k.cpu().float().numpy())

        return result
    
    def extract_keyword_boundary(self, wav):
        embeddings, feat_len = self.forward_audio(wav)
        result = self.cascaded_branch(embeddings, feat_len)
        return result

