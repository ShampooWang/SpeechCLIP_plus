from email.mime import audio
from math import ceil
import os
from os.path import exists
import sys
import shutil
import librosa
import logging
import numpy as np
import torch
import tqdm
import argparse
from avssl.module import S3prlSpeechEncoderPlus
from sklearn.cluster import KMeans
from torch.nn.utils.rnn import pad_sequence
from avssl.data import random_crop_max_length
import json

MAX_AUDIO_LEN = 102400

def parseArgs(argv):
    parser = argparse.ArgumentParser(description='Executing k-means to SSL feautures')
    parser.add_argument("--name", type=str)
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

        for _feat, _len in zip(_ssl_feats, _ssl_len):
            _feat = _feat[:_len]
            _feat = _feat.reshape(-1, _ssl_feats.shape[-1]).detach().cpu().numpy()
            _feat = [ np.squeeze(x) for x in np.split(_feat, len(_feat), axis=0) ]
            ssl_feats_list.extend(_feat)
        ssl_len_list.extend(_ssl_len.detach().cpu().tolist())

    # print("Saving SSl features...")
    # with open('ssl_feats.npy', "wb") as f:
    #     np.save(f, np.array(ssl_feats_list, dtype=object))
    print("Saving SSl feature length...")
    with open('ssl_len.npy', "wb") as f:
        np.save(f, np.array(ssl_len_list, dtype=object))
    # else:
    #     ssl_len_list = np.load('ssl_len.npy', allow_pickle=True)

    if not exists("cluster_ids.npy"):
        # Compute K-means Clustering
        print("Computing K-means")
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=1, verbose=1, random_state=0).fit(np.array(ssl_feats_list))
        cluster_ids = kmeans.labels_

        print("Saving cluster ids...")
        with open('cluster_ids.npy', "wb") as f:
            np.save(f, cluster_ids)
    else:
        cluster_ids = np.load("cluster_ids.npy", allow_pickle=True)
        # print(len(cluster_ids))
    assert len(ssl_len_list) == len(wav_files), f"{len(ssl_len_list)}, {len(wav_files)}"
    assert sum(ssl_len_list) == len(cluster_ids)

    cluster_label_dict = {}
    for i in range(len(ssl_len_list)):
        _wav_path = wav_files[i] 
        _start = 0 if i == 0 else sum(ssl_len_list[:i])
        _end = _start + ssl_len_list[:i]
        if len(ssl_len_list[_start:_end]) == 0: continue
        cluster_label_dict[_wav_path] = cluster_ids[_start:_end]
        
    assert len(cluster_label_dict) == len(ssl_len_list)

    cluster_label_json = json.dumps(cluster_label_dict)
    with open(args.output_file, "w") as f:
        f.write(cluster_label_json)




if __name__=="__main__":
    args = sys.argv[1:]
    main(args)
