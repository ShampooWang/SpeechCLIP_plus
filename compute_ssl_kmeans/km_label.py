from email.mime import audio
from math import ceil
import os
from os.path import exists
import sys
import shutil
import librosa
import logging
import numpy as np
import pickle
import torch
import tqdm
import argparse
from avssl.module import S3prlSpeechEncoderPlus
from sklearn.cluster import KMeans
from torch.nn.utils.rnn import pad_sequence
from avssl.data import random_crop_max_length
import json
from pytorch_lightning import seed_everything
import joblib

MAX_AUDIO_LEN = 102400
KM_PATH = "/mnt/md0/user_jeff/audio-visual-ssl/compute_ssl_kmeans/tests/hubert_base_ls960_L9_km500.km"
def parseArgs(argv):
    parser = argparse.ArgumentParser(description='Executing k-means to SSL feautures')
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", default=7122, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--feat_select_idx", default="last_hidden_state", type=str)
    parser.add_argument("--max_audio_len",  default=MAX_AUDIO_LEN, type=int)
    parser.add_argument("--input_file_list", type=str)
    parser.add_argument("--bsz", default=1, type=int)
    parser.add_argument("--n_clusters", default=100, type=int)
    parser.add_argument("--n_init", default=20, type=int)
    parser.add_argument("--output_file", type=str)

    return parser.parse_args(argv)

def process_wavs(wavs: list):
    wav_len = [ len(x) for x in wavs ]
    wavs = [
        random_crop_max_length(wavs[b], MAX_AUDIO_LEN, len(wavs[b]))
        for b in range(len(wavs))
    ]
    wavs = pad_sequence(wavs, batch_first=True)
    wav_len = [min(l, MAX_AUDIO_LEN) for l in wav_len]
    return wavs, wav_len

def main(argv):
    args = parseArgs(argv)
    seed_everything(args.seed)
    km_model = joblib.load(KM_PATH)
    # np.random.seed(args.seed) 
    # torch.manual_seed(args.seed)
    # km_model.random_state = args.seed
    C_np = torch.from_numpy(km_model.cluster_centers_).to(args.device)
    km_model.cluster_centers_ = km_model.cluster_centers_.astype(np.double)
    # print(C_np.shape)
    # exit()
    # Load input file
    print(f"Reading input file from {args.input_file_list}")
    wav_files = []
    with open(args.input_file_list, 'r') as f:
        for line in f:
            wav_files.append(line.strip("\n"))
    print(f"Found {len(wav_files)} Inputs!")

    # if not exists("ssl_len.npy"):
    # Get SSL features
    audio_encoder = S3prlSpeechEncoderPlus(**args.__dict__)
    audio_encoder.eval()
    bsz = args.bsz
    ssl_feats_list = []
    ssl_len_list = []
    cluster_label_dict = {}

    for i in tqdm.tqdm(range(0, len(wav_files), bsz)):
        if i + bsz >= len(wav_files) - 1:
            _wav_file_list = wav_files[i : ]
        else:
            _wav_file_list = wav_files[i : i + bsz]
        
        if len(_wav_file_list) == 0: continue

        _wav_list = []
        for _fp in _wav_file_list:
            _wav_list.append(
                torch.FloatTensor(librosa.load(_fp, sr=16_000)[0]).cuda()
            )
        _wavs, _wav_len = process_wavs(_wav_list)
        _ssl_feats, _ssl_len = audio_encoder(_wavs, _wav_len)

        for _wav_name, _feat, _len in zip(_wav_file_list, _ssl_feats, _ssl_len):
            _feat = _feat[:_len]
            _dist = torch.cdist(_feat, C_np)
            cluster_label_dict[_wav_name] = _dist.argmin(dim=-1).tolist()
            del _dist
    print(len(cluster_label_dict))
    cluster_label_json = json.dumps(cluster_label_dict)
    with open(args.output_file, "w") as f:
        f.write(cluster_label_json)




if __name__=="__main__":
    args = sys.argv[1:]
    main(args)
