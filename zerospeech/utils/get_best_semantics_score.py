import argparse
import sys
import os
import numpy as np
from collections import defaultdict

def main(path):
    result = defaultdict(lambda: defaultdict(list))
    for root, dirs, files in (os.walk(path)):
        for tmp_f in files:
            if tmp_f.endswith("dev_correlation.csv"):
                with open(os.path.join(root, tmp_f), "r") as open_f:
                    split_type = tmp_f.split("_")[2]
                    for i, line in enumerate(open_f):
                        if i > 0:
                            split_line = line.split(",")
                            dataset, score = split_line[0], split_line[-1]
                            result[split_type][dataset].append(float(score))
                            
            elif tmp_f.endswith("test_correlation.csv"):
                split_type = tmp_f.split("_")[2]
                with open(os.path.join(root, "score_semantic_test_pairs.csv"), "r") as weighted_f:
                    datasize_dict = defaultdict(int)
                    for i, line in enumerate(weighted_f):
                        if i > 0:
                            split_line = line.split(",")
                            datasize_dict[split_line[1]] += 1
                            
                with open(os.path.join(root, tmp_f), "r") as open_f:
                    tmp_score_dict = defaultdict(lambda: defaultdict(float))
                    for i, line in enumerate(open_f):
                        if i > 0:
                            split_line = line.split(",")
                            dataset, subset, score = split_line[0], split_line[1], split_line[-1]
                            tmp_score_dict[dataset][subset] = float(score)
                            
                    for dataset in tmp_score_dict:
                        result[f"{split_type}_unweighted"][dataset].append(sum(tmp_score_dict[dataset].values()) / len(tmp_score_dict[dataset].values()))
                        weighted_score = 0
                        for subset in tmp_score_dict[dataset]:
                            weighted_score += tmp_score_dict[dataset][subset] * datasize_dict[subset]
                        result[f"{split_type}_weighted"][dataset].append(weighted_score / sum(datasize_dict.values()))

    for split_type in result:
        for dataset in result[split_type]:
            print(f"{split_type}, {dataset}: {max(result[split_type][dataset])}")
    
if __name__ == "__main__":
    main(path = sys.argv[1])