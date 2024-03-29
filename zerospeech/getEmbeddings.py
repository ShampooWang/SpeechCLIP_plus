# from avssl.model import kwClip_plus as mymodels
# from avssl.model import kwClip as myoldmodels

# from avssl.model.kwClip_plus import *
from avssl.model.speechclip_plus import SpeechCLIP_plus
from avssl.module.speech_encoders_module import Custom_WavLM

import argparse
import os
import sys
import zerospeech_tasks
import logging
import random
import numpy as np
from pytorch_lightning import seed_everything
from typing import Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export BERT features from quantized units of audio files.')
    parser.add_argument("--wavlm", type=bool, default=False)
    parser.add_argument("--model_cls_name", type=str)
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

def loadModel(_cls, _path):
    _model = eval(_cls).load_from_checkpoint(_path)
    
    return _model

def load_wavlm(name: str):
    _model = Custom_WavLM(
            name=name, pretrained=True, max_audio_len=102400
    )
    return _model

def main(argv):
    args = parseArgs(argv)
    seed_everything(args.seed)

    task_cls = f"Task_{args.task_name}"

    assert hasattr(zerospeech_tasks,task_cls)
    if args.wavlm:
        logger.info(f"Using pretrained({args.model_cls_name}) wavlm")
        mymodel = load_wavlm(args.model_cls_name)
    else:
        # assert hasattr(mymodels,args.model_cls_name)
        logger.info(f"Loading model({args.model_cls_name}) from {args.model_ckpt}")
        mymodel = loadModel(args.model_cls_name, args.model_ckpt)
    mytask = getattr(zerospeech_tasks, task_cls)(**args.__dict__, my_model=mymodel)

    mytask.run(args.feat_select_idx)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)