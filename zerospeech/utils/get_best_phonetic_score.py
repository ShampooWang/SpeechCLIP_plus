import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import os
RESULT_ROOT = "/mnt/md0/user_jeff/zerospeech2021/result/phonetic"
SCORES = {}
MODELS = ["COCO_SpeechCLIP+", "COCO_SpeechCLIP_p2", "COCO_SpeechCLIP_c2"]
LAYER = [ i for i in range(16)]
for _model in MODELS:
    SCORES[_model] = {"test_clean_within":[], "test_clean_across":[], "test_other_within": [], "test_other_across": []}
    _root = os.path.join(os.path.join(RESULT_ROOT, _model))
    for _idx in LAYER:
        if _idx == 15 and _model != "COCO_h+_small_1e-5":
            continue
        _path = os.path.join(os.path.join(_root, f"hidden_state_{_idx}", "score_phonetic.csv"))
        with open(_path, "r") as _f:
            for i, _l in enumerate(_f):
                if i > 4:
                    _l = _l.strip("\n").split(",")
                    score = float(_l[-1])
                    score_type = ""
                    for i in range(3):
                        if i == 1:
                            score_type = score_type + ("_" + _l[i] + "_")
                        else:
                            score_type = score_type + _l[i]
                    SCORES[_model][score_type].append(score)
                    

for _model in MODELS:
    for _score_type in SCORES["COCO_SpeechCLIP+"].keys():
        score_list = np.array(SCORES[_model][_score_type])
        print(f"{_model}, {_score_type}: {np.min(score_list)}, idx: {np.argmin(score_list)}")