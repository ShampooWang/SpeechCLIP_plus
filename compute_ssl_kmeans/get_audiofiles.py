import os
from collections import defaultdict
import re
import json
from typing import Counter

DATASET_ROOT = "/mnt/md0/dataset/flickr"
SPLIT = ["train", "test", "dev"]
TEXT_FILE = "captions.txt"
CHAR = set()
special_char = ["(", "]", "=", "#", "-", ":", ")", "&"]
num_to_word = {"0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR", "5": "FIVE", "6": "SIX", "7": "SEVEN",
                "8": "EIGHT", "9": "NINE"}

special_sents = []
for sp in SPLIT:
    image_list_txt = os.path.join(
        DATASET_ROOT, f"Flickr_8k.{sp}Images.txt"
    )

    wav_base_path = os.path.join(DATASET_ROOT, "flickr_audio", "wavs")
    wav_list = os.listdir(wav_base_path)
    wav_names = {p[:-6] for p in wav_list if p.split(".")[-1] == "wav"}
    wav_names_to_paths = defaultdict(list)
    for p in wav_list:
        name = p.split("/")[-1][:-6]
        if name in wav_names:
            wav_names_to_paths[name].append(os.path.join(wav_base_path, p))

    assert TEXT_FILE in [
        "captions.txt",
        "Flickr8k.lemma.token.txt",
        "Flickr8k.token.txt",
    ], "Flickr8K text file must be one of them {}".format(
        ["captions.txt", "Flickr8k.lemma.token.txt", "Flickr8k.token.txt"]
    )
    caption_txt_path = os.path.join(DATASET_ROOT, TEXT_FILE)
    imageName2captions = {}

    if TEXT_FILE == "captions.txt":
        with open(caption_txt_path, "r") as f:
            for _l in f.readlines():
                # skip first line
                if _l.strip() == "image,caption":
                    continue

                _imgName, _caption = _l.split(".jpg,")
                assert isinstance(_imgName, str)
                assert isinstance(_caption, str)
                _caption = _caption.upper().strip()
                if _caption[-1] == ".":
                    _caption = _caption[:-1]
                    _caption = _caption.strip()
                if _imgName not in imageName2captions:
                    imageName2captions[_imgName] = []
                imageName2captions[_imgName].append(_caption)
    else:
        with open(caption_txt_path, "r") as f:
            for i, _line in enumerate(f.readlines()):
                _line = _line.strip()
                _out = re.split("#[0-9]", _line)
                assert len(_out) == 2, _line
                _imgName, _caption = re.split("#[0-9]", _line)
                _imgName = _imgName.replace(".jpg", "")
                _caption = _caption.strip()
                if _caption[-1] == ".":
                    _caption = _caption[:-1].strip()

                if _imgName not in imageName2captions:
                    imageName2captions[_imgName] = []
                imageName2captions[_imgName].append(_caption)

    id_pairs_path = os.path.join(DATASET_ROOT, "Flickr8k_idPairs.json")
    with open(id_pairs_path, "r") as f:
        _data = json.load(f)
        id2Filename = _data["id2Filename"]
        filename2Id = _data["filename2Id"]

    with open(image_list_txt, "r") as fp:
        file_list = []
        text = []
        for line in fp:
            line = line.strip()
            if line == "":
                continue

            image_name = line.split(".")[0]  # removed ".jpg"
            image_path = os.path.join(DATASET_ROOT, "Images", line)
            if image_name in wav_names:
                for p in wav_names_to_paths[image_name]:
                    _entry = {"id": filename2Id[image_name]}

                    if "txt" in os.path.basename(p).split("_")[-1].replace(
                        ".wav", ""
                    ):
                        continue

                    _subID = int(
                        os.path.basename(p).split("_")[-1].replace(".wav", "")
                    )

                    file_list.append(p)
                    sent = imageName2captions[image_name][_subID]
                    text.append(sent)

    # with open(f"/mnt/md0/dataset/flickr/audio_path_{sp}.txt", "w") as f:
    #     for fp in file_list:
    #         if fp == file_list[-1]:
    #             f.write(f"{fp}")
    #         else: 
    #             f.write(f"{fp}\n")
    with open(f"/mnt/md0/dataset/flickr/{sp}_audio_path_with_captions.txt", "w") as f:
        for _fp, _sent in zip(file_list, text):
            if _fp == file_list[-1]:
                f.write(f"{_fp}\t{_sent}")
            else: 
                f.write(f"{_fp}\t{_sent}\n")
# print(c)

CHAR = list(CHAR)
# print(CHAR)

