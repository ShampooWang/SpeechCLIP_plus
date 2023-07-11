from typing import List, Union

import clip
import editdistance as ed
import numpy as np
import sacrebleu
import torch
from clip.simple_tokenizer import SimpleTokenizer


def ter(hyps: List[Union[str, List[str]]], refs: List[Union[str, List[str]]]) -> float:
    """Token error rate calculator.

    Args:
        hyps (List[Union[str, List[str]]]): List of hypotheses.
        refs (List[Union[str, List[str]]]): List of references.

    Returns:
        float: Averaged token error rate overall utterances.
    """

    error_tokens = 0
    total_tokens = 0
    for h, r in zip(hyps, refs):
        error_tokens += ed.eval(h, r)
        total_tokens += len(r)
    return float(error_tokens) / float(total_tokens)


def wer(hyps: List[str], refs: List[str]) -> float:
    """Word error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged word error rate overall utterances.
    """

    hyps = [h.split(" ") for h in hyps]
    refs = [r.split(" ") for r in refs]
    return ter(hyps, refs)


def per(hyps: List[str], refs: List[str]) -> float:
    """Phoneme error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged phoneme error rate overall utterances.
    """

    return wer(hyps, refs)


def cer(hyps: List[str], refs: List[str]) -> float:
    """Character error rate calculator.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.

    Returns:
        float: Averaged character error rate overall utterances.
    """
    return ter(hyps, refs)


def report_bleu(hyps: List[str], refs: List[str]) -> None:
    """Computes and reports BLEU score.

    Args:
        hyps (List[str]): List of hypotheses.
        refs (List[str]): List of references.
    """

    print(sacrebleu.corpus_bleu(hyps, [refs]))


def cosine_semantics(results: list, dataset: str) -> list:
    cos_score = torch.load("/mnt/md0/user_jeff/Checkpoints/ViT-B32_cos_score.pt")
    cos_score = cos_score.cpu().detach().numpy()
    assert dataset in ["flickr", "coco"], dataset
    gt_tokens = np.load(
        f"/mnt/md0/user_jeff/audio-visual-ssl/avssl/data/{dataset}_stat/gt_tokens.npy",
        allow_pickle=True,
    )
    encoder = SimpleTokenizer().encoder

    cos_semantic_list = []
    for i, res in enumerate(results):
        text_toks = gt_tokens[i]
        score = 0
        counted_token = []
        for keywords in res["neighbors"].values():
            wordpeice = keywords[0][0]
            keywords_token = encoder[wordpeice]
            if keywords_token not in counted_token:
                counted_token.append(keywords_token)
                score += np.max(cos_score[keywords_token, text_toks])
        score = score / len(res["neighbors"])
        cos_semantic_list.append(score)
    del cos_score
    del gt_tokens

    return cos_semantic_list
