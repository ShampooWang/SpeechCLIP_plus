import argparse
import logging

import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, random_split

from ..base import OrderedNamespace
from ..data import FlickrDataset, PlacesImageCaptionDataset, collate_general
from ..model import (
    ParallelClipTextImage,
    ParallelSpeechClip_AttPool,
    ParallelSpeechClip_AttPool_FineGrain,
    ParallelSpeechClip_AttPool_FineGrainHookResBlk,
    ParallelSpeechClip_AttPoolText,
    ParallelSpeechClip_MeanPool,
    ParallelSpeechClip_MeanPool_Text,
)
from .base_task import BaseTask, TrainSpeechClipBaseTask


class TrainParallelSpeechClipBase(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self, model_cls):
        super().run(model_cls)


class TrainParallelSpeechClip_MeanPool(TrainParallelSpeechClipBase):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(ParallelSpeechClip_MeanPool)


class TrainParallelSpeechClip_AttPool(TrainParallelSpeechClipBase):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(ParallelSpeechClip_AttPool)


class TrainParallelSpeechClip_AttPool_FineGrain(TrainParallelSpeechClipBase):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(ParallelSpeechClip_AttPool_FineGrain)


class TrainParallelSpeechClip_AttPool_FineGrainHookResBlk(TrainParallelSpeechClipBase):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(ParallelSpeechClip_AttPool_FineGrainHookResBlk)


class TrainParallelSpeechClipTextBase(BaseTask):
    def __init__(self):
        super().__init__()

    def add_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--config", type=str, default=str, help="Configuration file."
        )
        parser.add_argument(
            "--ckpt", type=str, default="", help="Checkpoint to resume training."
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="Device name, could be cuda:X or cpu.",
        )
        parser.add_argument(
            "--gpus", type=int, default=1, help="Number of GPUs for training."
        )
        parser.add_argument("--njobs", type=int, default=0, help="Number of workers.")
        parser.add_argument("--seed", type=int, default=7122, help="Fix random seed.")
        parser.add_argument(
            "--save_path", type=str, default="", help="Directory to save ckpts."
        )
        parser.add_argument(
            "--train", action="store_true", default=False, help="Train model"
        )
        parser.add_argument(
            "--eval",
            action="store_true",
            default=False,
            help="Evaluate model on dev set",
        )
        parser.add_argument(
            "--test",
            action="store_true",
            default=False,
            help="Inference model on test set",
        )

        return parser

    def parse_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        args = parser.parse_args()

        if not torch.cuda.is_available():
            args.device = "cpu"
            args.gpus = 0

        self.args = args

        return args

    def run(self, model_cls, modalities=["text", "audio"]):
        assert self.args is not None

        seed_everything(self.args.seed)

        if self.args.ckpt != "":
            model = model_cls.load_from_checkpoint(self.args.ckpt).to(self.args.device)
            if self.args.save_path != "":
                model.config.save_path = self.args.save_path
            config = model.config
        else:
            self.args.ckpt = None
            config = yaml.load(open(self.args.config, "r"), Loader=yaml.FullLoader)
            config = OrderedNamespace([self.args, config])
            model = model_cls(config).to(config.device)
        self.config = config

        if config.data.dataset.name == "flickr":
            if self.args.train:
                tr_set = FlickrDataset(
                    split="train",
                    load_image=False,
                    modalities=modalities,
                    tokenizeText=False,
                    **config.data.dataset,
                )
            if self.args.train or self.args.eval:
                dv_set = FlickrDataset(
                    split="dev",
                    load_image=False,
                    modalities=modalities,
                    tokenizeText=False,
                    **config.data.dataset,
                )
            if self.args.test:
                test_set = FlickrDataset(
                    split="test",
                    load_image=False,
                    modalities=modalities,
                    tokenizeText=False,
                    **config.data.dataset,
                )
        elif config.data.dataset.name == "places":
            tr_set = PlacesImageCaptionDataset(
                split="train", load_image=False, **config.data.dataset
            )
            tr_len = int(len(tr_set) * config.data.split_ratio)
            tr_set, dv_set = random_split(
                tr_set,
                [tr_len, len(tr_set) - tr_len],
                generator=torch.Generator().manual_seed(config.seed),
            )
        else:
            raise NotImplementedError(f"Unknown dataset {config.data.dataset.name}")

        if self.args.train:
            tr_loader = DataLoader(
                tr_set,
                batch_size=config.data.batch_size,
                shuffle=True,
                num_workers=config.njobs,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_general,
            )
        if not hasattr(config.data, "dev_batch_size"):
            config.data.dev_batch_size = 4
        if self.args.train or self.args.eval:
            dv_loader = DataLoader(
                dv_set,
                batch_size=config.data.dev_batch_size,
                shuffle=False,
                num_workers=config.njobs,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_general,
            )
        if self.args.test:
            test_loader = DataLoader(
                test_set,
                batch_size=config.data.dev_batch_size,
                shuffle=False,
                num_workers=config.njobs,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_general,
            )

        if config.save_path != "":
            config.trainer.default_root_dir = config.save_path

        model_checkpoint_val_loss = ModelCheckpoint(
            dirpath=config.trainer.default_root_dir,
            filename="{epoch}-{step}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            every_n_epochs=1,
            save_last=True,
        )

        model_checkpoint_recall = ModelCheckpoint(
            dirpath=config.trainer.default_root_dir,
            filename="{epoch}-{step}-{val_recall_mean_1:.4f}",
            monitor="val_recall_mean_1",
            save_top_k=3,
            mode="max",
            every_n_epochs=1,
        )

        trainer = Trainer(
            callbacks=[
                TQDMProgressBar(),
                model_checkpoint_val_loss,
                model_checkpoint_recall,
            ],
            enable_progress_bar=True,
            gpus=config.gpus,
            **config.trainer,
        )

        if self.args.train:
            trainer.fit(model, tr_loader, dv_loader, ckpt_path=config.ckpt)
        if self.args.eval:
            trainer.validate(model, dv_loader, ckpt_path=config.ckpt)
        if self.args.test:
            trainer.validate(model, test_loader, ckpt_path=config.ckpt)


class TrainParallelSpeechClip_MeanPool_Text(TrainParallelSpeechClipTextBase):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(ParallelSpeechClip_MeanPool_Text)


class TrainParallelSpeechClip_AttPool_Text(TrainParallelSpeechClipTextBase):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(ParallelSpeechClip_AttPoolText)


class OriginalCLIPTextImage(TrainParallelSpeechClipTextBase):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(ParallelClipTextImage, modalities=["text", "image"])
