import json
from avssl.model import SpeechCLIP_plus
from avssl.util import compute_cos_alignments
import sys

def main(model_path, detok_result_path, file_path):
    with open(detok_result_path, "r") as j:
        results = json.load(j)
    model = SpeechCLIP_plus.load_from_checkpoint(model_path)
    scores = compute_cos_alignments(model, results, file_path)
    print(sum(scores) / len(scores))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])