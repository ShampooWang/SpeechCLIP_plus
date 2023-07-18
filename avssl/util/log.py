import argparse
import logging
from typing import Union

from pytorch_lightning.loggers import WandbLogger

from ..base import OrderedNamespace


def set_logging(args: argparse.Namespace) -> None:
    """Setup logging.

    Args:
        args (argparse.Namespace): Arguments.
    """

    level = getattr(logging, str(args.log_level).upper())
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s",
        datefmt="%m-%d %H:%M",
    )


def set_pl_logger(config: OrderedNamespace):
    """Setup PyTorch Lightning logger.

    Args:
        config (OrderedNamespace): configurations.

    Returns:
        Union[bool, LightningLoggerBase]: Logger.
    """

    # ddp issue: https://github.com/Lightning-AI/lightning/issues/13166

    logger_type = config.trainer.get("logger", None)
    project = config.logger.project

    if logger_type is None or not config.train:
        return None
    elif isinstance(logger_type, bool):
        return logger_type
    elif logger_type == "wandb" or isinstance(logger_type, WandbLogger):
        name = config.trainer.default_root_dir.split("/")[-1]
        logger = WandbLogger(
            project=project, name=name, save_dir=config.trainer.default_root_dir
        )
        if hasattr(logger.experiment.config, "update"):
            logger.experiment.config.update(
                config.to_dict(),
            )
        return logger
    else:
        raise NotImplementedError(f"Unknown logger type = {logger_type}")
