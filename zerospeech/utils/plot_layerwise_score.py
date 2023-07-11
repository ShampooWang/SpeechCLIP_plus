import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import os
RESULT_ROOT = "/mnt/md0/user_jeff/zerospeech2021/result"
SCORES = {}
LAYER = [str(i) for i in range(15)]
for _model in ["wavlm", "COCO_SpeechCLIP_p+"]:
    SCORES[_model] = {"Lib":[], "Syn":[]}
    _root = os.path.join(os.path.join(RESULT_ROOT, _model))
    for _idx in LAYER:
        if int(_idx) > 12 and _model == "wavlm":
            for _t in SCORES[_model]:
                SCORES[_model][_t].append(0)
            continue
        _dir = os.path.join(os.path.join(_root, f"hidden_state_{_idx}", "score_semantic_dev_correlation.csv"))
        with open(_dir, "r") as _f:
            for i, _l in enumerate(_f):
                if i > 0:
                    _s = float(_l.strip("\n").split(",")[-1])
                    SCORES[_model]["Lib"].append(_s) if i == 1 else SCORES[_model]["Syn"].append(_s)




WIDTH = 0.2
PIC_ROOT_PATH = "/mnt/md0/user_jeff/zerospeech2021/pic"
label = ["CNN"] + [str(i) for i in range(1,13)] + ["Trm1", "Trm2"]
for _t in SCORES["wavlm"]:
    fig = plt.figure(figsize=(8,4))
    plt.title(f"ZeroSpeech2021 semantics dev ({_t})")
    x1 = list(range(0, len(LAYER), 1))
    x2 = [x + WIDTH for x in x1]
    print(f"{_t}, walvm, {max(SCORES['wavlm'][_t])}")
    print(f"{_t}, COCO_SpeechCLIP_p+, {max(SCORES['COCO_SpeechCLIP_p+'][_t])}")
    plt.bar(x1, SCORES["wavlm"][_t], label='WavLM Base+', width=WIDTH)
    plt.bar(x2, SCORES["COCO_SpeechCLIP_p+"][_t], label='Parallel SpeechCLIP+', width=WIDTH)

    plt.xlabel("layer", fontsize=12)
    plt.ylabel("correlation", fontsize=12)
    plt.xticks([x + WIDTH/2 for x in x1], label, fontsize=10)     #設定 X 軸刻度標籤
    # plt.xticks(x, values) # x的刻度值

    plt.legend(loc='lower right')
    plt.savefig(os.path.join(PIC_ROOT_PATH, f'z2021_semantics_{_t}.png'))
    # plt.close()