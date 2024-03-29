import logging
import types

from fairseq.models.hubert.hubert import HubertConfig, HubertModel
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder

logger = logging.getLogger(__name__)

import json
from typing import Dict, List, Optional, Tuple, Union

import fairseq
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.utils import index_put
from s3prl import hub
from s3prl.utility.download import _urls_to_filepaths
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoProcessor
from transformers.file_utils import copy_func

from ..util import freeze_model, init_weights, random_crop_max_length
from .wavlm_modules.modules import GradMultiply
from .wavlm_modules.WavLM import TransformerEncoder as WavLMTransformerEncoder
from .wavlm_modules.WavLM import WavLM, WavLMConfig
from .weighted_sum import WeightedSumLayer

FEAT_SELECT_IDX_WEIGHTED_SUM_MODE = "weighted_sum"


def custom_FairseqTransformerEncoder_extract_features(
    self, x, padding_mask=None, tgt_layer=None
):
    if padding_mask is not None:
        x = index_put(x, padding_mask, 0)

    x_conv = self.pos_conv(x.transpose(1, 2))
    x_conv = x_conv.transpose(1, 2)
    x = x + x_conv

    if not self.layer_norm_first:
        x = self.layer_norm(x)

    x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    layer_results = [x.transpose(0, 1)]
    r = None
    for i, layer in enumerate(self.layers):
        dropout_probability = np.random.random()
        if not self.training or (dropout_probability > self.layerdrop):
            x, _ = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
            layer_results.append(x.transpose(0, 1))
        if i == tgt_layer:
            r = x
            break

    if r is not None:
        x = r

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    return x, layer_results


def customFunc_hubert_forward(
    self,
    source: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    mask: bool = True,
    output_layer: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """output layer is 1-based"""
    features = self.forward_features(source)

    features = features.transpose(1, 2)
    features = self.layer_norm(features)
    unmasked_features = features.clone()

    if padding_mask is not None:
        padding_mask = self.forward_padding_mask(features, padding_mask)

    if self.post_extract_proj is not None:
        features = self.post_extract_proj(features)

    features = self.dropout_input(features)
    unmasked_features = self.dropout_features(unmasked_features)

    if mask:
        x, mask_indices = self.apply_mask(features, padding_mask, None)
    else:
        x = features
        mask_indices = None

    x, layer_results = self.encoder(
        x,
        padding_mask=padding_mask,
        layer=None if output_layer is None else output_layer - 1,
    )

    return {"x": x, "layer_results": layer_results}


def customFunc_wavlm_forward(
    self,
    source: torch.Tensor,
    padding_mask: Optional[torch.Tensor] = None,
    output_layer: Optional[int] = None,
    mask: bool = False,
) -> Dict[str, torch.Tensor]:
    if self.feature_grad_mult > 0:
        features = self.feature_extractor(source)
        if self.feature_grad_mult != 1.0:
            features = GradMultiply.apply(features, self.feature_grad_mult)
    else:
        with torch.no_grad():
            features = self.feature_extractor(source)

    features = features.transpose(1, 2)
    features = self.layer_norm(features)

    if padding_mask is not None:
        padding_mask = self.forward_padding_mask(features, padding_mask)

    if self.post_extract_proj is not None:
        features = self.post_extract_proj(features)

    features = self.dropout_input(features)

    if mask:
        x, mask_indices = self.apply_mask(features, padding_mask)
    else:
        x = features

    x, layer_results = self.encoder(
        x,
        padding_mask=padding_mask,
        layer=None if output_layer is None else output_layer - 1,
    )

    return {"x": x, "layer_results": layer_results}


def custom_WavLMTransformerEncoder_extract_features(
    self, x, padding_mask=None, streaming_mask=None, tgt_layer=None
):
    if padding_mask is not None:
        x[padding_mask] = 0

    x_conv = self.pos_conv(x.transpose(1, 2))
    x_conv = x_conv.transpose(1, 2)
    x = x + x_conv

    if not self.layer_norm_first:
        x = self.layer_norm(x)

    x = F.dropout(x, p=self.dropout, training=self.training)

    # B x T x C -> T x B x C
    x = x.transpose(0, 1)

    layer_results = [x.transpose(0, 1)]
    z = None
    r = None
    pos_bias = None
    for i, layer in enumerate(self.layers):
        dropout_probability = np.random.random()
        if not self.training or (dropout_probability > self.layerdrop):
            x, z, pos_bias = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_weights=False,
                self_attn_mask=streaming_mask,
                pos_bias=pos_bias,
            )
            layer_results.append(x.transpose(0, 1))
        if i == tgt_layer:
            r = x
            break

    if r is not None:
        x = r

    # T x B x C -> B x T x C
    x = x.transpose(0, 1)

    return x, layer_results


class FairseqSpeechEncoder_Hubert(nn.Module):
    MODEL2URL = {
        "hubert": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
        "hubert_base": "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt",
        "hubert_large_ll60k": "https://dl.fbaipublicfiles.com/hubert/hubert_large_ll60k.pt",
    }

    MODEL_DOWNSAMPLE_RATE = {
        "hubert": 320,
        "hubert_base": 320,
        "hubert_large_ll60k": 320,
    }

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        trainable: bool = False,
        feat_select_idx: Union[str, list] = "all",
        layer_drop: Union[str, float] = 0.0,
        max_audio_len: int = -1,
        reinit_layers: List[int] = [],
        unfreeze_layers: List[int] = [],
        normalize_hiddenstates: bool = False,
        normalize_type: str = "s3prl",
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.pretrained = pretrained
        self.trainable = trainable
        self.feat_select_idx = feat_select_idx
        self.max_audio_len = max_audio_len
        self.reinit_layers = reinit_layers
        self.unfreeze_layers = unfreeze_layers
        self.normalize_hiddenstates = normalize_hiddenstates
        self.layer_drop = layer_drop
        assert normalize_type in ["s3prl", "method1", "method2"], normalize_type
        if self.normalize_hiddenstates:
            logger.info("Normalize hidden states ({})".format(normalize_type))
        self.normalize_type = normalize_type

        self.general_model_preprocessing()

    def general_model_preprocessing(self):
        if self.name.startswith("hubert"):
            assert self.name in self.MODEL2URL, "Model name({}) should be in {}".format(
                self.name, self.MODEL2URL.keys()
            )
            ckpt = _urls_to_filepaths(self.MODEL2URL[self.name], refresh=False)
            self.apply_customHubertForward()  # modify Hubert Functions for extracting hidden states
            model, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [ckpt]
            )
            self.encoder = model[0]
            self.encoder_task = task
            self.config = self.encoder_task.cfg

        elif self.name.startswith("wavlm"):
            assert hasattr(
                self, "MODEL2PATH"
            ), f"You have to specify your own wavlm checkpoints' locations in MODEL2PATH"
            assert (
                self.name in self.MODEL2PATH
            ), "Model name({}) should be in {}".format(
                self.name, self.MODEL2PATH.keys()
            )
            ckpt = torch.load(self.MODEL2PATH[self.name])
            self.config = WavLMConfig(ckpt["cfg"])
            self.encoder = WavLM(self.config)
            self.encoder.load_state_dict(ckpt["model"])
            self.apply_customWavLMForward()
        else:
            raise NotImplementedError(self.name)

        logger.info(f"Normalize waveform = ({self.config.normalize})")

        if hasattr(self.encoder, "get_downsample_rates"):
            self.downsample_rate = self.encoder.get_downsample_rates("hidden_states")
        else:
            self.downsample_rate = self.MODEL_DOWNSAMPLE_RATE[self.name]

        if not self.pretrained:
            self.encoder.apply(init_weights)

        if not self.trainable:
            freeze_model(self.encoder)
            self.encoder.eval()

        if self.name.startswith("hubert") or self.name.startswith("wavlm"):
            if (
                isinstance(self.layer_drop, float)
                and self.layer_drop >= 0.0
                and self.layer_drop <= 1.0
            ):
                self.encoder.encoder.layerdrop = self.layer_drop
            elif self.layer_drop == "original":
                pass
            else:
                raise ValueError(f"layer_drop = {self.layer_drop} is not supported.")

            assert not (len(self.reinit_layers) > 0 and len(self.unfreeze_layers) > 0)
            if len(self.reinit_layers) > 0:
                logger.warning(f"Reinitializing encoder layers {self.reinit_layers}")
                assert self.trainable
                for i, layer in enumerate(self.encoder.encoder.layers):
                    if i in self.reinit_layers:
                        layer.apply(init_weights)
                    else:
                        freeze_model(layer)

                freeze_model(self.encoder.encoder.pos_conv)
                freeze_model(self.encoder.layer_norm)
                freeze_model(self.encoder.feature_extractor)
                freeze_model(self.encoder.post_extract_proj)
                self.encoder.feature_grad_mult = 0

            if len(self.unfreeze_layers) > 0:
                logger.warning(
                    f"Freezing encoder layers excluding {self.unfreeze_layers}"
                )
                assert self.trainable
                for i, layer in enumerate(self.encoder.encoder.layers):
                    if i in self.unfreeze_layers:
                        pass
                    else:
                        freeze_model(layer)

                freeze_model(self.encoder.encoder.pos_conv)
                freeze_model(self.encoder.layer_norm)
                freeze_model(self.encoder.feature_extractor)
                freeze_model(self.encoder.post_extract_proj)
                self.encoder.feature_grad_mult = 0

        self.out_dim = 0
        with torch.no_grad():
            wav = [torch.randn(16000, dtype=torch.float, device="cpu")]
            padded_wav, wav_padding_mask = self.preprocess_input(wavs=wav)
            output = self.custom_forward(padded_wav, wav_padding_mask)
            self.upstream_model_hiddenstates_len = len(output["layer_results"])
            self.out_dim = output["x"].shape[2]

        logger.info(
            f"Loaded speech encoder ({self.name}): out_dim = {self.out_dim} layer_drop = {self.encoder.encoder.layerdrop}"
        )

        if self.feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            logger.info(
                f"Using weighted sum for all hiddenstates({self.upstream_model_hiddenstates_len})"
            )
            assert self.upstream_model_hiddenstates_len > 0

            self.weightedsum_layer = WeightedSumLayer(
                n_weights=self.upstream_model_hiddenstates_len,
                normalize_features=self.normalize_hiddenstates
                and self.normalize_type == "s3prl",
            )

    def trainable_params(self) -> list:
        if self.trainable and len(self.reinit_layers) == 0:
            return list(self.parameters())
        if self.trainable and len(self.reinit_layers) > 0:
            params = []
            for i in self.reinit_layers:
                params += list(self.encoder.encoder.layers[i].parameters())
            if not self.encoder.encoder.layer_norm_first:
                params += list(self.encoder.encoder.layer_norm.parameters())
            return params
        else:
            if self.feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
                logger.info("Adding weightedsum params")
                params = list(self.weightedsum_layer.parameters())
                return params
            else:
                return []

    def apply_customHubertForward(self):
        # add method
        TransformerEncoder.extract_features = copy_func(
            custom_FairseqTransformerEncoder_extract_features
        )
        # add method
        HubertModel.customHubertForward = copy_func(customFunc_hubert_forward)

    def preprocess_input(self, wavs):
        if self.config.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        return padded_wav, wav_padding_mask

    def custom_forward(self, wav, pad_mask):
        if hasattr(self.encoder, "customHubertForward"):
            features = self.encoder.customHubertForward(
                wav,
                padding_mask=pad_mask,
                mask=None,
            )
        elif hasattr(self.encoder, "customFuncWavLMforward"):
            features = self.encoder.customFuncWavLMforward(
                wav,
                padding_mask=pad_mask,
                mask=None,
            )
        else:
            raise NotImplementedError()

        return features

    def forward(
        self,
        batch: dict,
        feat_select_idx: Union[str, list] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[Union[torch.Tensor, list], torch.Tensor]:
        wav, wav_len = batch["wav"], batch["wav_len"]
        if isinstance(wav, torch.Tensor):
            if wav.dim() == 2:
                if len(wav_len) > 0:
                    wav = [wav[b, : wav_len[b]] for b in range(len(wav))]
                else:
                    wav = [wav[b] for b in range(len(wav))]
            elif wav.dim() == 1:
                wav = [wav]

        padded_wav, wav_padding_mask = self.preprocess_input(wavs=wav)

        if self.trainable:
            features = self.custom_forward(padded_wav, wav_padding_mask)
        else:
            with torch.no_grad():
                features = self.custom_forward(padded_wav, wav_padding_mask)

        if self.normalize_hiddenstates:
            if self.normalize_type.startswith("method"):
                for i in range(len(features["layer_results"])):
                    if self.normalize_type == "method1":
                        # method1
                        features["layer_results"][i] = features["layer_results"][i] / (
                            torch.norm(
                                features["layer_results"][i], dim=-1, keepdim=True
                            )
                            + 1e-8
                        )
                    elif self.normalize_type == "method2":
                        # method2
                        features["layer_results"][i] = features["layer_results"][
                            i
                        ] / torch.mean(
                            torch.norm(features["layer_results"][i], dim=-1), dim=-1
                        ).view(
                            -1, 1, 1
                        )
                    else:
                        # s3prl
                        stacked_feature = F.layer_norm(
                            stacked_feature, (stacked_feature.shape[-1],)
                        )

        feat = {
            "last_hidden_state": features["layer_results"][-1],
            "hidden_states": tuple(features["layer_results"]),
        }
        wav_len = [len(w) for w in wav]

        feat_len = (
            torch.LongTensor([round(l / self.downsample_rate) for l in wav_len])
            .type_as(feat["last_hidden_state"])
            .long()
            .clip(max=feat["last_hidden_state"].shape[1])
        )

        if feat_select_idx is None:
            feat_select_idx = self.feat_select_idx

        return_list = []
        if feat_select_idx == "all":
            return_list.extend([feat, feat_len])
        elif feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            return_list.extend(
                [self.weightedsum_layer(feat["hidden_states"]), feat_len]
            )
        elif isinstance(feat_select_idx, list):
            feat = [feat["hidden_states"][i] for i in feat_select_idx]
            return_list.extend([feat, feat_len])
        elif feat_select_idx in feat:
            return_list.extend([feat[feat_select_idx], feat_len])
        else:
            raise KeyError(feat_select_idx)

        if return_hidden_states:
            return_list.append(feat["hidden_states"])

        return tuple(return_list)


class Custom_WavLM(FairseqSpeechEncoder_Hubert):
    MODEL2PATH = {
        "wavlm_base": "/mnt/md0/user_jeff/Checkpoints/wavlm_pt/WavLM-Base.pt",
        "wavlm_base_plus": "/mnt/md0/user_jeff/Checkpoints/wavlm_pt/WavLM-Base+.pt",
        "wavlm_large": "/mnt/md0/user_jeff/Checkpoints/wavlm_pt/WavLM-Large.pt",
    }

    MODEL_DOWNSAMPLE_RATE = {
        "wavlm_base": 320,
        "wavlm_base_plus": 320,
        "wavlm_large": 320,
    }

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        trainable: bool = False,
        feat_select_idx: Union[str, list] = "all",
        layer_drop: Union[str, float] = 0.0,
        max_audio_len: int = -1,
        reinit_layers: List[int] = [],
        unfreeze_layers: List[int] = [],
        normalize_hiddenstates: bool = False,
        normalize_type: str = "s3prl",
        **kwargs,
    ):
        super().__init__(
            name,
            pretrained,
            trainable,
            feat_select_idx,
            layer_drop,
            max_audio_len,
            reinit_layers,
            unfreeze_layers,
            normalize_hiddenstates,
        )

    def apply_customWavLMForward(self):
        # add method
        WavLMTransformerEncoder.extract_features = copy_func(
            custom_WavLMTransformerEncoder_extract_features
        )
        # add method
        WavLM.customFuncWavLMforward = copy_func(customFunc_wavlm_forward)


class S3prlSpeechEncoderPlus(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        trainable: bool = False,
        device: str = "cpu",
        feat_select_idx: Union[str, list] = "all",
        layer_drop: Union[str, float] = 0.0,
        max_audio_len: int = -1,
        reinit_layers: List[int] = [],
        unfreeze_layers: List[int] = [],
        **kwargs,
    ):
        super().__init__()

        self.name = name
        self.pretrained = pretrained
        self.trainable = trainable
        self.device = device
        self.feat_select_idx = feat_select_idx
        self.max_audio_len = max_audio_len
        self.reinit_layers = reinit_layers
        self.unfreeze_layers = unfreeze_layers
        self.layer_drop = layer_drop

        self.encoder = getattr(hub, name)().to(device)
        if hasattr(self.encoder, "get_downsample_rates"):
            self.downsample_rate = self.encoder.get_downsample_rates("hidden_states")
        else:
            self.downsample_rate = 160

        if not pretrained:
            self.encoder.apply(init_weights)

        if not trainable:
            freeze_model(self.encoder)

        self.general_model_preprocessing()

    def custom_forward(self, wav):
        output = self.encoder(wav)
        return {
            "x": output["last_hidden_state"],
            "layer_results": output["hidden_states"],
        }

    def forward(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        feat_select_idx: Union[str, list] = None,
        return_hidden_states: bool = False,
    ) -> Tuple[Union[torch.Tensor, list], torch.Tensor]:
        if isinstance(wav, torch.Tensor):
            if wav.dim() == 2:
                if len(wav_len) > 0:
                    wav = [wav[b, : wav_len[b]] for b in range(len(wav))]
                else:
                    wav = [wav[b] for b in range(len(wav))]
            elif wav.dim() == 1:
                wav = [wav]

        if self.trainable:
            feat = self.encoder(wav)
        else:
            with torch.no_grad():
                feat = self.encoder(wav)

        wav_len = [len(w) for w in wav]

        feat_len = (
            torch.LongTensor([int(l / self.downsample_rate) for l in wav_len])
            .clip(max=feat["hidden_states"][0].shape[1])
            .to(feat["last_hidden_state"].device)
        )

        if feat_select_idx is None:
            feat_select_idx = self.feat_select_idx

        return_list = []
        if feat_select_idx == "all":
            return_list = [feat, feat_len]
        elif feat_select_idx == FEAT_SELECT_IDX_WEIGHTED_SUM_MODE:
            return_list = [self.weightedsum_layer(feat["hidden_states"]), feat_len]
        elif isinstance(feat_select_idx, list):
            feat = [feat["hidden_states"][i] for i in feat_select_idx]
            return_list = [feat, feat_len]
        elif feat_select_idx in feat:
            return_list = [feat[feat_select_idx], feat_len]
        else:
            raise KeyError(feat_select_idx)

        if return_hidden_states:
            return_list.append(feat["hidden_states"])

        return tuple(return_list)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return self

    def feature_extractor_zerospeech(self, wav):
        result = []
        embeddings, feat_len = self.forward(wav)
        for _embs, _len in zip(embeddings, feat_len):
            result.append(_embs[:_len].cpu().float().numpy())
        return result
