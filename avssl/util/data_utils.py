from collections import defaultdict

import json
import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from clip.simple_tokenizer import SimpleTokenizer
from scipy.optimize import linear_sum_assignment


def clip_fp_alignment(batch, maxLen):
    new_fp_ali_list = []
    new_segment_num_list = []
    for ali in batch["fp_alignment"]:
        new_ali = ali[ali.nonzero()].squeeze(-1).clip(max=maxLen).unique()
        new_fp_ali_list.append(new_ali)
        new_segment_num_list.append(len(new_ali))

    new_fp_alignment = pad_sequence(new_fp_ali_list, batch_first=True).to(
        batch["fp_alignment"].device
    )
    new_segment_num = torch.LongTensor(new_segment_num_list).to(
        batch["fp_alignment"].device
    )
    batch.update({"fp_alignment": new_fp_alignment, "segment_num": new_segment_num})


def fix_random_crop_alginment(crop_idx, fp_alignment, segment_num, maxLen):
    if crop_idx is None:
        return fp_alignment, segment_num
    else:
        crop_idx = round(crop_idx / 50)
        maxFpLen = round(maxLen / 320)

    modified_ali = list(
        filter(
            lambda x: x - 1 >= crop_idx and x - 1 < crop_idx + maxFpLen, fp_alignment
        )
    )
    modified_ali.append(crop_idx + maxFpLen)
    modified_ali = torch.LongTensor(modified_ali).to(fp_alignment.device) - crop_idx
    modified_ali = modified_ali[modified_ali.nonzero()].squeeze(-1).unique()

    return modified_ali, len(modified_ali)


def random_crop_max_length(
    wav: torch.Tensor, max_len: int, orig_len: int = 1000000000
) -> torch.Tensor:
    """Randomly crop an audio feature sequence into max_len.

    Args:
        audio (torch.Tensor): Audio features (T, *)
        max_len (int): Maximum length.
        orig_len (int, optional): Original length of audio. Defaults to 1000000000.

    Returns:
        torch.Tensor: Cropped audio features.
    """
    audio_len = min(len(wav), orig_len)
    if audio_len <= max_len or max_len < 0:
        return wav[:audio_len]

    offset = np.random.randint(audio_len - max_len)
    return wav[offset : offset + max_len]


def get_keypadding_mask(max_length: int, data_lens: torch.Tensor) -> torch.Tensor:
    bsz = data_lens.shape[0]
    key_padding_mask = torch.ones([bsz, max_length])
    for mask, len in zip(key_padding_mask, data_lens):
        mask[:len] = 0.0
    key_padding_mask = key_padding_mask.type_as(data_lens).bool()

    return key_padding_mask


def compute_dynamic_keyword_neighbors(
    model,
    K: int,
    retreival_type: str,
    outputs: dict,
    tokenEmbeddings: torch.Tensor,
    keyword_embeddings_list: list,
    gold_texts: list,
    feat_len_list: list,
    emb_pinv=None,
):
    hit_rate_list = []
    all_retok_outputs = []
    batch_size = model.config.data.dev_batch_size
    for b_idx, i in zip(
        range(len(outputs)),
        range(0, len(gold_texts), batch_size),
    ):
        _gold_texts = gold_texts[i : i + batch_size]
        _feat_len_list = feat_len_list[i : i + batch_size]
        gold_subword_toks_set = [
            set(model.clip.tokenizer.encode(_text)) for _text in _gold_texts
        ]

        b_instance_idx = 0
        for keyword_embeddings in keyword_embeddings_list[b_idx]:
            _bsz, _max_feat_len = (
                keyword_embeddings.shape[0],
                keyword_embeddings.shape[1],
            )

            if retreival_type == "pseudo_inverse":
                assert (
                    emb_pinv is not None
                ), f"You are using pseudo-inverse to retreive the keywords, please provide pseudo-inverse"
                kw_retrevial_score = (
                    emb_pinv.float()
                    @ keyword_embeddings.view(-1, model.subword_embd_dim)
                    .float()
                    .reshape(-1, model.subword_embd_dim)
                    .permute(1, 0)
                ).permute(1, 0)
            else:
                kw_retrevial_score = F.cosine_similarity(
                    keyword_embeddings.view(-1, model.subword_embd_dim, 1),
                    tokenEmbeddings.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            _k_values, _k_indices = torch.topk(kw_retrevial_score, K)

            assert _k_values.shape == (
                _bsz * _max_feat_len,
                K,
            ), _k_values.shape
            _k_indices = _k_indices.view(_bsz, _max_feat_len, K)
            _k_values = _k_values.view(_bsz, _max_feat_len, K)

            for x in range(_bsz):
                _hit_rate = 0
                hit_kw = []
                tmp_outputs = {}
                _feat_len = _feat_len_list[b_instance_idx + x]
                _gold_subword_toks_set = gold_subword_toks_set[b_instance_idx + x]
                for _keyword_i in range(_feat_len):
                    tmp_outputs["keyword_{}".format(_keyword_i)] = []

                    # check if nearest K subword appears in gold text
                    top_k_toks = set(
                        [
                            model.clip.reducedl2Original[_ind.item()]
                            if model.clip.selected_text_emb_ids is not None
                            else _ind.item()
                            for _ind in _k_indices[x, _keyword_i]
                        ]
                    )

                    if bool(top_k_toks & _gold_subword_toks_set):
                        _hit_rate += 1 / _feat_len
                        hit_token_id = int(list(top_k_toks & _gold_subword_toks_set)[0])
                        hit_token = model.clip.tokenizer.decoder[
                            hit_token_id
                            if model.clip.selected_text_emb_ids is not None
                            else model.clip.reducedl2Original[hit_token_id]
                        ]
                        hit_kw.append(hit_token)

                    for _ind, _dist in zip(
                        _k_indices[x, _keyword_i], _k_values[x, _keyword_i]
                    ):
                        tmp_outputs["keyword_{}".format(_keyword_i)].append(
                            [
                                model.clip.tokenizer.decoder[
                                    model.clip.reducedl2Original[_ind.item()]
                                    if model.clip.selected_text_emb_ids is not None
                                    else _ind.item()
                                ],
                                _dist.item(),
                            ]
                        )

                hit_rate_list.append(_hit_rate)

                all_retok_outputs.append(
                    {
                        "gold": _gold_texts[b_instance_idx + x],
                        "neighbors": tmp_outputs,
                        "hit_kw": hit_kw,
                        "kw_hit_rate": _hit_rate,
                    }
                )
            b_instance_idx += _bsz

    return hit_rate_list, all_retok_outputs


def compute_fixed_keyword_neighbors(
    model,
    K: int,
    retreival_type: str,
    tokenEmbeddings: torch.Tensor,
    all_keyword_embeddings: torch.Tensor,
    gold_texts: list,
    emb_pinv=None,
):
    hit_rate_list = [0] * model.keyword_num
    kw_top_ret = [[] for _ in range(model.keyword_num)]
    all_retok_outputs = []
    batch_size = model.config.data.dev_batch_size

    for i in tqdm(range(0, len(gold_texts) + batch_size, batch_size)):
        _gold_texts = gold_texts[i : i + batch_size]
        _bsz = len(_gold_texts)
        if len(_gold_texts) == 0:
            break

        gold_subword_toks_set = [
            set(model.clip.tokenizer.encode(_text)) for _text in _gold_texts
        ]

        if retreival_type == "pseudo_inverse":
            assert (
                emb_pinv is not None
            ), f"You are using pseudo-inverse to retreive the keywords, please provide pseudo-inverse"
            kw_retrevial_score = (
                emb_pinv.float()
                @ all_keyword_embeddings[i : i + _bsz]
                .view(-1, model.subword_embd_dim)
                .float()
                .reshape(-1, model.subword_embd_dim)
                .permute(1, 0)
            ).permute(1, 0)
        else:
            kw_retrevial_score = F.cosine_similarity(
                all_keyword_embeddings[i : i + _bsz].view(
                    -1, model.subword_embd_dim, 1
                ),
                tokenEmbeddings.transpose(0, 1).unsqueeze(0),
                dim=1,
            )
        _k_values, _k_indices = torch.topk(kw_retrevial_score, K)

        assert _k_values.shape == (
            _bsz * model.keyword_num,
            K,
        ), _k_values.shape
        _k_indices = _k_indices.view(_bsz, model.keyword_num, K)
        _k_values = _k_values.view(_bsz, model.keyword_num, K)

        for x in range(_bsz):
            tmp_outputs = {}
            for _keyword_i in range(model.keyword_num):
                tmp_outputs["keyword_{}".format(_keyword_i)] = []

                # check if nearest K subword appears in gold text
                top_k_toks = set(
                    [
                        model.clip.reducedl2Original[_ind.item()]
                        if model.clip.selected_text_emb_ids is not None
                        else _ind.item()
                        for _ind in _k_indices[x, _keyword_i]
                    ]
                )
                if bool(top_k_toks & gold_subword_toks_set[x]):
                    hit_rate_list[_keyword_i] += 1
                    hit_token_id = int(list(top_k_toks & gold_subword_toks_set[x])[0])
                    kw_top_ret[_keyword_i].append(hit_token_id)

                for _ind, _dist in zip(
                    _k_indices[x, _keyword_i], _k_values[x, _keyword_i]
                ):
                    tmp_outputs["keyword_{}".format(_keyword_i)].append(
                        [
                            model.clip.tokenizer.decoder[
                                model.clip.reducedl2Original[_ind.item()]
                                if model.clip.selected_text_emb_ids is not None
                                else _ind.item()
                            ],
                            _dist.item(),
                        ]
                    )

            all_retok_outputs.append(
                {
                    "gold": gold_texts[x],
                    "neighbors": tmp_outputs,
                }
            )

    return hit_rate_list, kw_top_ret, all_retok_outputs

def compute_cos_alignments(model, all_retok_outputs, file_path):
    ENCODER, DECODER = SimpleTokenizer().encoder, SimpleTokenizer().decoder
    scores = []
    output = []

    for tokenize_ouput in tqdm(all_retok_outputs):
        gold_toks = SimpleTokenizer().encode(text=tokenize_ouput["gold"])
        gold_toks = [ model.clip.original2Reduced[tok] for tok in gold_toks[1:gold_toks.index(49407)] ]
        gold_toks = torch.LongTensor(gold_toks).to(model.device)
        
        all_pred_toks = []
        for neighbor in tokenize_ouput["neighbors"].values():
            all_pred_toks += [ ENCODER[kw[0]] for kw in neighbor ]
        all_pred_toks = [ model.clip.original2Reduced[tok] for tok in all_pred_toks ]
        all_pred_toks = torch.LongTensor(list(set(all_pred_toks))).to(model.device)

        gold_embd, pred_embd = model.clip.model.token_embedding(gold_toks), model.clip.model.token_embedding(all_pred_toks)
        pairwise_cos_similarity = F.normalize(gold_embd, dim=-1) @ F.normalize(pred_embd, dim=-1).T
        pairwise_cos_similarity = pairwise_cos_similarity.cpu().detach().numpy()

        row_ind, col_ind = linear_sum_assignment(pairwise_cos_similarity, maximize=True)
        assign_best_score_idx = np.argsort(pairwise_cos_similarity[row_ind, col_ind], axis=0)[::-1][:model.keyword_num]
        topk_gold_toks = [ model.clip.reducedl2Original[tok.item()] for tok in gold_toks[row_ind[assign_best_score_idx]] ]
        topk_pred_toks = [ model.clip.reducedl2Original[tok.item()] for tok in all_pred_toks[col_ind[assign_best_score_idx]] ]
        topk_gold_text = [ DECODER[tok] for tok in topk_gold_toks ]
        topk_pred_text = [ DECODER[tok] for tok in topk_pred_toks ]
        topk_scores = pairwise_cos_similarity[row_ind, col_ind][assign_best_score_idx]
        scores.append(topk_scores.mean(0))

        output.append(
            {
                "gold": tokenize_ouput["gold"],
                f"top{model.keyword_num}_gold_subwords": topk_gold_text,
                f"top{model.keyword_num}_pred_subwords": topk_pred_text,
                f"top{model.keyword_num}_scores": topk_scores.tolist(),
            }
        )

    with open(file_path, "w") as f:
        json.dump(output, f, indent=4)

    return scores





        

