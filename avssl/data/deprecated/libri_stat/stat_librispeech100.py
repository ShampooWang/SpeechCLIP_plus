import os
import re

import clip
import numpy as np
import torch
import tqdm
from clip.simple_tokenizer import SimpleTokenizer

DATASET_DIR = "/work/vjsalt22/dataset/LibriSpeech100"

# TXT_FILE = "Flickr8k.lemma.token.txt"
TXT_FILE = "text/output.txt"
# TXT_FILE = "captions.txt"
CHAR = [
    "a ",
    "b ",
    "c ",
    "d ",
    "e ",
    "f ",
    "g ",
    "h ",
    "i ",
    "j ",
    "k ",
    "l ",
    "m ",
    "n ",
    "o ",
    "p ",
    "q ",
    "r " "s ",
    "t ",
    "u ",
    "v ",
    "w ",
    "x ",
    "y ",
    "z ",
]

if __name__ == "__main__":
    flickr_stat = np.load(
        "/work/vjsalt22/hsuanfu/audio-visual-ssl/avssl/data/flickr_stat/text_clip_vocab_usage_byfreq.npy"
    )
    if not os.path.exists("id_freq.npy"):
        print("start a new session")
        with open(os.path.join(DATASET_DIR, TXT_FILE), "r") as f:
            _data = f.readlines()

        captions = []
        for i, _line in enumerate(tqdm.tqdm(_data)):
            _line = _line.strip()
            captions.append(_line)

        tokens = clip.tokenize(texts=captions, context_length=77, truncate=True)

        tokens = tokens.flatten().numpy()

        unique, counts = np.unique(tokens, return_counts=True)

        result_arr = np.asarray((unique, counts)).T
        np.save(
            "/work/vjsalt22/hsuanfu/audio-visual-ssl/avssl/data/libri_stat/id_freq.npy",
            result_arr,
        )
    else:
        result_arr = np.load("id_freq.npy")
        unique, counts = result_arr[:, 0], result_arr[:, -1]

    if os.path.exists("mutual_token_list.npy"):
        mutual_token_list = np.load("mutual_token_list.npy")

        total_freq = 0
        mutual_freq = []
        for token, freq in zip(unique, counts):
            if (token != 0) & (token != 49406) & (token != 49407):
                total_freq += freq
                if token in flickr_stat[:, 0]:
                    mutual_freq.append(freq)

        coverage = 0
        for token, freq in tqdm.tqdm(zip(mutual_token_list, mutual_freq)):
            if token not in CHAR:
                coverage += freq
        print(f"total token numbers: {total_freq}")
        print(f"Mutual token numbers w/o characters: {coverage}")
        print(f"coverage: {coverage / total_freq}")
    else:
        total_token_num = 0
        coverage = 0
        mutual_id_list = []
        for token, freq in zip(unique, counts):
            if (token != 0) & (token != 49406) & (token != 49407):
                total_token_num += freq
                if token in flickr_stat[:, 0]:
                    coverage += freq
                    mutual_id_list.append([token])

        np.save("mutual_id_list.npy", mutual_id_list)
        print(f"total token numbers: {total_token_num}")
        print(f"Mutual token numbers: {coverage}")
        print(f"coverage: {coverage / total_token_num}")
        if os.path.exists("mutual_id_freq_list.npy"):
            id_freq = np.load("mutual_id_freq_list.npy")
            id_list, freq_list = id_freq[:, 0], id_freq[:, -1]
            mutual_token_list = []
            for id, freq in tqdm.tqdm(zip(id_list, freq_list)):
                mutual_token_list.append([SimpleTokenizer().decode([id]), freq])
            print(mutual_token_list)
            np.save("mutual_token_list.npy", mutual_token_list)
