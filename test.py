import json
from avssl.model import SpeechCLIP_plus
from avssl.util import compute_cos_alignments

with open("/work/jgtf0322/SpeechCLIP_plus/checkpoints/slt_ckpts/SpeechCLIP/base/Flickr/cascaded/test/retokenizeText/keywords_ep0.json", "r") as j:
    results = json.load(j)
model = SpeechCLIP_plus.load_from_checkpoint("/work/jgtf0322/SpeechCLIP_plus/checkpoints/slt_ckpts/SpeechCLIP/base/Flickr/cascaded/epoch_58-step_6902-val_recall_mean_1_7.7700.ckpt")
scores = compute_cos_alignments(model, results)
print(sum(scores) / len(scores))

