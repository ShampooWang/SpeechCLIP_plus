import argparse
import logging

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, random_split

from ..base import OrderedNamespace
from ..data import (
    FlickrImageCaptionDataset,
    PlacesImageCaptionDataset,
    collate_image_captions,
)
from ..model import (
    KeywordCascadedSpeechClip,
    KeywordCascadedSpeechClip_ProjVQ,
    KeywordCascadedSpeechClip_ProjVQ_Cosine,
    VQCascadedSpeechClip,
)
from .base_task import BaseTask, TrainSpeechClipBaseTask


class TrainVQCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(VQCascadedSpeechClip)


class TrainKeywordCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KeywordCascadedSpeechClip)


class TrainKeywordProjVQCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KeywordCascadedSpeechClip_ProjVQ)


class TrainKeywordProjVQCosineCascadedSpeechClip(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KeywordCascadedSpeechClip_ProjVQ_Cosine)