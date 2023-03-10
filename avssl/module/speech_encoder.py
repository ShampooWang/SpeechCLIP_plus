import logging
from typing import List, Tuple, Union

import torch
from s3prl import hub
from torch import nn

from ..data import random_crop_max_length
from ..util import freeze_model, init_weights


class S3prlSpeechEncoder(nn.Module):
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
        """Speech Encoder with S3PRL (v0.3.1)

        Args:
            name (str): Name of speech encoder.
            pretrained (bool, optional): Init with pretrained model. Defaults to False.
            trainable (bool, optional): Whether to update the model while training. Defaults to False.
            device (str, optional): Device. Defaults to "cpu".
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to "all".
            layerdrop (Union[str, float], optional): Layer drop rate. Defaults to 0.0.
        """
        super().__init__()

        self.name = name
        self.pretrained = pretrained
        self.trainable = trainable
        self.device = device
        self.feat_select_idx = feat_select_idx
        self.max_audio_len = max_audio_len
        self.reinit_layers = reinit_layers
        self.unfreeze_layers = unfreeze_layers

        self.encoder = getattr(hub, name)().to(device)
        if hasattr(self.encoder, "get_downsample_rates"):
            self.downsample_rate = self.encoder.get_downsample_rates("hidden_states")
        else:
            self.downsample_rate = 160

        if not pretrained:
            self.encoder.apply(init_weights)

        if not trainable:
            freeze_model(self.encoder)

        if self.name.startswith("hubert"):
            if (
                isinstance(layer_drop, float)
                and layer_drop >= 0.0
                and layer_drop <= 1.0
            ):
                self.encoder.model.encoder.layerdrop = layer_drop
            elif layer_drop == "original":
                pass
            else:
                raise ValueError(f"layer_drop = {layer_drop} is not supported.")

            assert not (len(reinit_layers) > 0 and len(unfreeze_layers) > 0)
            if len(reinit_layers) > 0:
                logging.warning(f"Reinitializing encoder layers {reinit_layers}")
                assert self.trainable
                for i, layer in enumerate(self.encoder.model.encoder.layers):
                    if i in reinit_layers:
                        layer.apply(init_weights)
                    else:
                        freeze_model(layer)

                freeze_model(self.encoder.model.encoder.pos_conv)
                freeze_model(self.encoder.model.layer_norm)
                freeze_model(self.encoder.model.feature_extractor)
                freeze_model(self.encoder.model.post_extract_proj)
                self.encoder.model.feature_grad_mult = 0
            if len(unfreeze_layers) > 0:
                logging.warning(f"Freezing encoder layers excluding {unfreeze_layers}")
                assert self.trainable
                for i, layer in enumerate(self.encoder.model.encoder.layers):
                    if i in unfreeze_layers:
                        pass
                        # layer.apply(init_weights)
                    else:
                        freeze_model(layer)

                freeze_model(self.encoder.model.encoder.pos_conv)
                freeze_model(self.encoder.model.layer_norm)
                freeze_model(self.encoder.model.feature_extractor)
                freeze_model(self.encoder.model.post_extract_proj)
                self.encoder.model.feature_grad_mult = 0

        self.out_dim = 0
        with torch.no_grad():
            wav = [torch.randn(16000, dtype=torch.float, device=device)]
            feat = self.encoder(wav)
            self.out_dim = feat["last_hidden_state"].shape[2]

        logging.info(
            f"Loaded s3prl speech encoder ({name}): out_dim = {self.out_dim} layer_drop = {self.encoder.model.encoder.layerdrop}"
        )

    def trainable_params(self) -> list:
        if self.trainable and len(self.reinit_layers) == 0:
            return list(self.parameters())
        if self.trainable and len(self.reinit_layers) > 0:
            params = []
            for i in self.reinit_layers:
                params += list(self.encoder.model.encoder.layers[i].parameters())
            if not self.encoder.model.encoder.layer_norm_first:
                params += list(self.encoder.model.encoder.layer_norm.parameters())
            return params
        else:
            return []

    def forward(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        feat_select_idx: Union[str, list] = None,
    ) -> Tuple[Union[torch.Tensor, list], torch.Tensor]:
        """Forward function for S3PRL speech encoder

        Args:
            wav (Union[torch.Tensor, list]): List of waveforms. (L, )
            wav_len (Union[torch.Tensor, list]): List of waveforms' lengths. Defaults to [].
            feat_select_idx (Union[str, list], optional): Feature selection indices. Defaults to None.

        Raises:
            KeyError: feat_select_idx is not "all", "hidden_states",
                      "last_hidden_state", or list.

        Returns:
            Tuple[Union[torch.Tensor, list], torch.Tensor]: Hidden features and their lengths.
        """

        if isinstance(wav, torch.Tensor):
            if wav.dim() == 2:
                if len(wav_len) > 0:
                    wav = [wav[b, : wav_len[b]] for b in range(len(wav))]
                else:
                    wav = [wav[b] for b in range(len(wav))]
            elif wav.dim() == 1:
                wav = [wav]

        if self.training:
            wav = [
                random_crop_max_length(wav[b], self.max_audio_len, len(wav[b]))
                for b in range(len(wav))
            ]

        if self.trainable:
            feat = self.encoder(wav)
        else:
            with torch.no_grad():
                feat = self.encoder(wav)

        # if len(wav_len) == 0:
        wav_len = [len(w) for w in wav]

        feat_len = torch.LongTensor(
            [round(l / self.downsample_rate) for l in wav_len]
        ).to(feat["last_hidden_state"].device)
        # feat_len = torch.clamp_max(feat_len, feat["last_hidden_state"].shape[1])
        feat_len = torch.clamp_max(feat_len, feat["hidden_states"][0].shape[1])

        if feat_select_idx is None:
            feat_select_idx = self.feat_select_idx

        if feat_select_idx == "all":
            return feat, feat_len
        elif isinstance(feat_select_idx, list):
            feat = [feat["hidden_states"][i] for i in feat_select_idx]
            return feat, feat_len
        elif feat_select_idx in feat:
            return feat[feat_select_idx], feat_len
        else:
            raise KeyError(feat_select_idx)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return self
