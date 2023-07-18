from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from fairseq.utils import index_put

from .wavlm_modules.modules import GradMultiply


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
