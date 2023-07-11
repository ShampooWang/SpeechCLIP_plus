import os
import re

import clip
import numpy as np
import sentencepiece as spm
import torch
import tqdm
from clip.simple_tokenizer import SimpleTokenizer

DATASET_DIR = "/work/vjsalt22/dataset/LibriSpeech100"

# TXT_FILE = "Flickr8k.lemma.token.txt"
TXT_FILE = "text/output.txt"
# TXT_FILE = "captions.txt"

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    sp.load(
        "/work/vjsalt22/hsuanfu/audio-visual-ssl/tmp_tools/s3prl/s3prl/downstream/ctc/vocab/subword-300.model"
    )

    # Gets all tokens as Python list.
    vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
    # print(vocabs)
    # Aggregates the frequency of each token in the training data.
    freq = {}
    with open("/work/vjsalt22/dataset/LibriSpeech100/text/output.txt", "r") as f:
        for line in f:
            line = line.rstrip()
            for piece in sp.encode_as_pieces(line):
                freq.setdefault(piece, 0)
                freq[piece] += 1
    print(len(freq))
    mutual_token = np.load("mutual_token_uppercase.npy")
    print(len(mutual_token))
    # for i in range(len(mutual_token)):
    #     mutual_token[i] = mutual_token[i].upper()
    # with open('mutual_token_list.txt', 'w') as f:
    #     for i in mutual_token:
    #         f.write(i)
    #     f.close()

    # print(mutual_token)
    coverage = 0
    for token in freq:
        if token in mutual_token:
            coverage += freq[token]
    print(coverage / sum(freq.values()))
