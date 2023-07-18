import argparse
import sys
import os
import numpy as np
from collections import defaultdict

def main(path):
    result = defaultdict(lambda: defaultdict(list))
    for root, dirs, files in (os.walk(path)):
        for tmp_f in files:
            if tmp_f.endswith(".csv"):
                with open(os.path.join(root, tmp_f), "r") as open_f:
                    for i, line in enumerate(open_f):
                        if i > 0:
                            split_line = line.split(",")
                            dataset, subset, score_type, score = split_line[0], split_line[1], split_line[2], split_line[3]
                            result[f"{dataset}_{subset}"][score_type].append(float(score))
    
    for dataset in result:
        for score_type in result[dataset]:
            print(f"{dataset}, {score_type}: {min(result[dataset][score_type])}")
    
if __name__ == "__main__":
    main(path = sys.argv[1])