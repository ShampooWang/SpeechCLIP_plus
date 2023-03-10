import json
import logging
import os
import re
from collections import defaultdict
from typing import List

from .base_dataset import BaseDataset, BaseImageCaptionDataset


class CoCoDataset(BaseDataset):
    def __init__(
        self,
        dataset_root: str,
        modalities: List,
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
        load_audio: bool = True,
        load_image: bool = True,
        wav_rm_silence: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset_root=dataset_root,
            split=split,
            image_transform=image_transform,
            audio_transform=audio_transform,
            target_sr=target_sr,
            load_audio=load_audio,
            load_image=load_image,
            **kwargs,
        )

        assert len(modalities) > 0, "Dataset's modalities cannot be none"
        self.modalities = modalities

        assert self.split in ["train", "val"]

        data_json_path = os.path.join(
            self.dataset_root, "SpokenCOCO", f"SpokenCOCO_{self.split}.json"
        )
        with open(data_json_path, "r") as f:
            raw_data = json.load(f)["data"]

        for _entry in raw_data:
            if "audio" in self.modalities or "text" in self.modalities:
                for _capion in _entry["captions"]:
                    _ent_data = {
                        "id": int(_entry["image"].split("_")[-1].replace(".jpg", "")),
                    }

                    if "audio" in self.modalities:
                        _ent_data["wav"] = os.path.join(
                            self.dataset_root, "SpokenCOCO", _capion["wav"]
                        )
                    if "image" in self.modalities:
                        _ent_data["image"] = os.path.join(
                            self.dataset_root, "mscoco_img", _entry["image"]
                        )
                    if "text" in self.modalities:
                        _ent_data["text"] = _capion["text"]
                    self.data.append(_ent_data)
            else:
                self.data.append(
                    {
                        "image": os.path.join(
                            self.dataset_root, "mscoco_img", _entry["image"]
                        ),
                        "id": int(_entry["image"].split("_")[-1].replace(".jpg", "")),
                    }
                )

        logging.info(f"SpokenCOCO ({self.split}): {len(self.data)} samples")
