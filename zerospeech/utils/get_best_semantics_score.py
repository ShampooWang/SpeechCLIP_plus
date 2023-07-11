import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import os
RESULT_ROOT = "/mnt/md0/user_jeff/zerospeech2021/result"
SCORES = {}
LAYER = [str(i) for i in range(15)]
for _model in ["coco_p+_reproduce", "topline_wo_attn_lr2e-5_cos_wo_dcl"]:
    SCORES[_model] = {"lib":[], "syn":[]}
    _root = os.path.join(os.path.join(RESULT_ROOT, _model))
    for _idx in LAYER:
        if int(_idx) > 12 and _model == "topline_wo_attn_lr2e-5_cos_wo_dcl":
            for _t in SCORES[_model]:
                SCORES[_model][_t].append(0)
            continue
        # elif int(_idx) == 15 and (_model == "Flickr_SpeechCLIP_c+" or _model == "Flickr_SpeechCLIP_p+" or _model == "COCO_SpeechCLIP_p+" or _model == "COCO_SpeechCLIP_c+" or _model == "COCO_SpeechCLIP_p2_hubert" or _model == "COCO_SpeechCLIP_c2_hubert"):
        #     for _t in SCORES[_model]:
        #         SCORES[_model][_t].append(0)
            continue
        _dir = os.path.join(os.path.join(_root, f"hidden_state_{_idx}", "score_semantic_dev_correlation.csv"))
        with open(_dir, "r") as _f:
            for i, _l in enumerate(_f):
                if i > 0:
                    _s = float(_l.strip("\n").split(",")[-1])
                    SCORES[_model]["lib"].append(_s) if i == 1 else SCORES[_model]["syn"].append(_s)





for _t in ["lib", "syn"]:
    print(f"{_t}, topline_wo_attn_lr2e-5_cos_wo_dcl, {max(SCORES['topline_wo_attn_lr2e-5_cos_wo_dcl'][_t])}, layer: {np.argmax(np.array(SCORES['topline_wo_attn_lr2e-5_cos_wo_dcl'][_t]))}")
    print(f"{_t}, coco_p+_reproduce, {max(SCORES['coco_p+_reproduce'][_t])}, layer: {np.argmax(np.array(SCORES['coco_p+_reproduce'][_t]))}")