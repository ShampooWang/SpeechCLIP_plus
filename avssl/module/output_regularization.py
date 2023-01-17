import numpy as np
import torch
import torch.nn as nn

from ..util import freeze_model
from .speech_encoder_plus import Custom_WavLM

# device = torch.device("cuda")
# ORIG_WAVLM = Custom_WavLM(name="wavlm_base_plus",
#             pretrained=True,
#             trainable=False,
#             feat_select_idx="weighted_sum",
#             layer_drop=0.0,
#             max_audio_len=102400)
# ORIG_WAVLM = nn.DataParallel(ORIG_WAVLM)
# ORIG_WAVLM = ORIG_WAVLM.to(device)
# replicas = nn.parallel.replicate(ORIG_WAVLM, )
# ORIG_WAVLM.eval()


class Audio_encoder_regularization(nn.Module):
    def __init__(
        self, audio_encoder_config, init_weight=1, criterion=nn.MSELoss()
    ) -> None:
        super().__init__()
        self.ORIG_WAVLM = Custom_WavLM(
            name="wavlm_base_plus",
            pretrained=True,
            trainable=False,
            feat_select_idx="last_hidden_state",
            layer_drop=0.0,
            max_audio_len=102400,
        )
        self.criterion = criterion
        self.weight = init_weight

    def forward(self, wav, wav_len, hidden_states):
        target_states, _, _ = self.ORIG_WAVLM(wav, wav_len, return_hidden_states=True)
        assert (
            hidden_states.shape == target_states.shape
        ), f"{hidden_states.shape}, {target_states.shape}"

        return self.criterion(hidden_states, target_states)
