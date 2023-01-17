from avssl.model import kwClip_plus as mymodels
from avssl.module.speech_encoder_plus import *
from avssl.model.kwClip_plus import *

import argparse
import os
import sys
import zerospeech_tasks
import logging
import random
import numpy as np
from pytorch_lightning import seed_everything

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export BERT features from quantized units of audio files.')
    parser.add_argument("--s3prl", type=bool, default=False)
    parser.add_argument("--model_cls_name",type=str)
    parser.add_argument("--model_ckpt",type=str)
    parser.add_argument("--task_input_dir",type=str,default="/mnt/md0/dataset/zerospeech2021/semantic/")
    parser.add_argument("--task_name",type=str,default="semantic")
    parser.add_argument("--output_result_dir",type=str)
    parser.add_argument("--seed", type=int, default=7122)
    parser.add_argument("--inference_bsz", type=int, default=64)
    parser.add_argument("--feat_select_idx", type=str)
    parser.add_argument(
            "--run_dev", action="store_true", default=False, help="run dev"
        )
    parser.add_argument(
            "--run_test", action="store_true", default=False, help="run dev"
        )
    return parser.parse_args(argv)

def loadModel(_cls, _path, feat_select_idx=None):
    _model = getattr(mymodels,_cls).load_from_checkpoint(_path)
    if feat_select_idx is not None:
        if feat_select_idx != "cif":
            feat_select_idx = int(feat_select_idx)
            _model.audio_encoder.feat_select_idx = feat_select_idx
    return _model

def load_s3prl(name: str, feat_select_idx: Union[str, list]):
    _model = S3prlSpeechEncoderPlus(
            name=name, pretrained=True, feat_select_idx=feat_select_idx, max_audio_len=102400
    )
    return _model

def main(argv):
    args = parseArgs(argv)
    seed_everything(args.seed)

    task_cls = f"Task_{args.task_name}"

    assert hasattr(zerospeech_tasks,task_cls)
    if args.s3prl:
        logger.info(f"Loading model({args.model_cls_name}) from s3prl")
        mymodel = load_s3prl(args.model_cls_name, args.feat_select_idx)
    else:
        assert hasattr(mymodels,args.model_cls_name)
        logger.info(f"Loading model({args.model_cls_name}) from {args.model_ckpt}")
        mymodel = loadModel(args.model_cls_name, args.model_ckpt, args.feat_select_idx)
    mytask = getattr(zerospeech_tasks, task_cls)(**args.__dict__, my_model=mymodel)

    mytask.run()

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)