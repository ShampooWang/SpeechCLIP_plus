import torch
from torch import nn
import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import clip
from clip.simple_tokenizer import SimpleTokenizer
import pickle as pl
from os.path import exists
import regex as re
import ftfy
import html

gt_tokens = []
with open("/mnt/md0/user_jeff/audio-visual-ssl/exp/wavlm/COCO_c2_large/retokenizeText/keywords_ep0.json", "r") as j:
    results = json.load(j)
    for res in tqdm.tqdm(results):
        text_toks = SimpleTokenizer().encode(text=res["gold"])
        text_toks = text_toks[1:text_toks.index(49407)]
        gt_tokens.append(text_toks)
        
with open("./coco_stat/gt_tokens.npy", "wb") as fp:
    np.save(fp, np.array(gt_tokens))