import abc
import argparse
import os

import pytorch_lightning
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, random_split

from ..base import OrderedNamespace
from ..data import (
    CoCoDataset,
    FlickrDataset,
    PlacesImageCaptionDataset,
    collate_general,
)
from ..util import add_general_arguments, set_logging, set_pl_logger


def gradient_norm(model, check_nan=True):
    total_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            if check_nan:
                if torch.isnan(param_norm):
                    print(name)
                    exit()
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        print(f"my_model/grad_norm: {gradient_norm(model)}")


class BaseTask:
    def __init__(self):
        self.args = None
        self.config = None

    @abc.abstractmethod
    def add_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        raise NotImplementedError

    @abc.abstractmethod
    def parse_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError


class TrainSpeechClipBaseTask(BaseTask):
    def __init__(self):
        super().__init__()

    def add_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_general_arguments(parser)
        return parser

    def parse_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        args = parser.parse_args()

        if not torch.cuda.is_available():
            args.device = "cpu"
            args.gpus = 0

        self.args = args
        set_logging(args)

        return args

    def run(self, model_cls, custom_trainer_callbacks=[]):
        assert self.args is not None

        seed_everything(self.args.seed)

        if self.args.resume != "":
            self.args.ckpt = self.args.resume

        if self.args.ckpt != "":
            model = model_cls.load_from_checkpoint(self.args.ckpt)
            if self.args.save_path != "":
                model.config.save_path = self.args.save_path
            config = model.config
        else:
            self.args.ckpt = None
            config = yaml.load(open(self.args.config, "r"), Loader=yaml.FullLoader)
            config = OrderedNamespace([self.args, config])
            model = model_cls(config)

        if not hasattr(config.data.dataset, "modalities"):
            config.data.dataset.modalities = ["audio", "image", "text"]

        self.config = config

        if config.data.dataset.name == "flickr":
            if self.args.train:
                tr_set = FlickrDataset(
                    split="train",
                    # load_image=False,
                    # tokenizeText=False,
                    # modalities=["audio", "image", "text"],
                    audioEncoderDownsamplingRate=getattr(config.audio_encoder, "downsampling_rate", 320),
                    **config.data.dataset,
                )
            if self.args.train or self.args.eval:
                dv_set = FlickrDataset(
                    split="dev",
                    # load_image=False,
                    # tokenizeText=False,
                    # modalities=["audio", "image", "text"],
                    audioEncoder_downsamplingRate=getattr(config.audio_encoder, "downsampling_rate", 320),
                    **config.data.dataset,
                )
            if self.args.test:
                test_set = FlickrDataset(
                    split="test",
                    # load_image=False,
                    # tokenizeText=False,
                    # modalities=["audio", "image", "text"],
                    audioEncoder_downsamplingRate=getattr(config.audio_encoder, "downsampling_rate", 320)
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
        elif config.data.dataset.name == "coco":
            if self.args.train:
                tr_set = CoCoDataset(
                    split="train",
                    # modalities=["audio", "image", "text"],
                    **config.data.dataset,
                )
            if self.args.train or self.args.eval:
                dv_set = CoCoDataset(
                    split="val",
                    # modalities=["audio", "image", "text"],
                    **config.data.dataset,
                )
            if self.args.test:
                test_set = CoCoDataset(
                    split="test",
                    # modalities=["audio", "image", "text"],
                    **config.data.dataset,
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
            config.data.dev_batch_size = config.data.batch_size

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
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
            save_last=True,
        )

        model_checkpoint_recall = ModelCheckpoint(
            dirpath=config.trainer.default_root_dir,
            filename="{epoch}-{step}-{val_recall_mean_10:.4f}",
            monitor="val_recall_mean_10",
            save_top_k=1,
            mode="max",
            every_n_epochs=1,
        )

        model_checkpoint_kw_hit_rate = ModelCheckpoint(
            dirpath=config.trainer.default_root_dir,
            filename="{epoch}-{step}-{val_kw_hit_rate:.4f}",
            monitor="val_kw_hit_rate",
            save_top_k=1,
            mode="max",
            every_n_epochs=1,
        )

        model_checkpoint_cos_semantics = ModelCheckpoint(
            dirpath=config.trainer.default_root_dir,
            filename="{epoch}-{step}-{val_kw_hit_rate:.4f}",
            monitor="val_cos_semantics",
            save_top_k=1,
            mode="max",
            every_n_epochs=1,
        )

        config.trainer.logger = set_pl_logger(
            config,
        )

        config.gpus = self.args.gpus

        save_kw_hit_rate = config.log_setting.get("save_kw_hit_rate", False)
        if save_kw_hit_rate and config.model_settings.cascaded_objective_weight > 0:
            trainer = Trainer(
                callbacks=[
                    TQDMProgressBar(),
                    model_checkpoint_val_loss,
                    model_checkpoint_recall,
                    model_checkpoint_kw_hit_rate,
                    # model_checkpoint_cos_semantics,
                    *custom_trainer_callbacks,
                ],
                enable_progress_bar=True,
                gpus=config.gpus,
                resume_from_checkpoint=None
                if self.args.resume == ""
                else self.args.resume,
                **config.trainer,
            )
        else:
            trainer = Trainer(
                callbacks=[
                    TQDMProgressBar(),
                    model_checkpoint_val_loss,
                    model_checkpoint_recall,
                    *custom_trainer_callbacks,
                ],
                enable_progress_bar=True,
                gpus=config.gpus,
                resume_from_checkpoint=None
                if self.args.resume == ""
                else self.args.resume,
                **config.trainer,
            )

        if self.args.train:
            trainer.fit(model, tr_loader, dv_loader, ckpt_path=self.args.ckpt)
        if self.args.eval:
            trainer.validate(model, dv_loader, ckpt_path=config.ckpt, verbose=True)
        if self.args.test:
            trainer.validate(model, test_loader, ckpt_path=config.ckpt)
