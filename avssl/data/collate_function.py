from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_image_captions(batch: Tuple):
    if len(batch[0]) == 2:
        # no id given
        audio_list, audio_len_list, image_list = [], [], []
        for audio, image in batch:
            audio_list.append(audio)
            audio_len_list.append(len(audio))
            image_list.append(image)

        return audio_list, audio_len_list, image_list

    elif len(batch[0]) == 3:
        # id given
        audio_list, audio_len_list, image_list, id_list = [], [], [], []
        for audio, image, id in batch:
            audio_list.append(audio)
            audio_len_list.append(len(audio))
            image_list.append(image)
            id_list.append(id)

        return audio_list, audio_len_list, image_list, id_list

    else:
        raise NotImplementedError("Data format no implemented in collator")

    # # general code
    # return_batch = [ [] for _ in batch[0] ]
    # for _data in batch:
    #     for i, feat in enumerate(_data):
    #         return_batch[i].append(feat)
    # return return_batch


def collate_general(batch: Tuple):
    keysInBatch = list(batch[0].keys())
    if "wav" in keysInBatch and isinstance(batch[0]["wav"], torch.Tensor):
        keysInBatch.append("wav_len")
    return_dict = {k: [] for k in keysInBatch}
    for _row in batch:
        for _key in keysInBatch:
            # assert _key in _row.keys(), f"_key: {_key}, keys: {_row.keys()}"
            if _key == "wav_len":
                return_dict[_key].append(len(_row["wav"]))
            else:
                assert _key in _row.keys(), f"_key: {_key}, keys: {_row.keys()}"
                return_dict[_key].append(_row[_key])

    for key in return_dict:
        if isinstance(return_dict[key][0], torch.Tensor):
            if key == "wav":
                return_dict[key] = pad_sequence(return_dict[key], batch_first=True)
            else:
                return_dict[key] = torch.stack(return_dict[key], dim=0)
        else:
            if key == "alignments":
                max_len = max([len(_ali) for _ali in return_dict[key]])
                for i in range(len(return_dict[key])):
                    _len = len(return_dict[key][i])
                    return_dict[key][i] = return_dict[key][i] + [[-1, -1]] * (
                        max_len - _len
                    )
                return_dict[key] = torch.LongTensor(return_dict[key])
            else:
                return_dict[key] = torch.LongTensor(return_dict[key])

    return return_dict
