import os
import sys
import json
import argparse
# from black import out
import progressbar
from pathlib import Path
from time import time
import numpy as np
import librosa

import torch

# from utils.utils_functions import writeArgs, loadRobertaCheckpoint
from avssl.model import (
    KeywordCascadedSpeechClip,
    KeywordCascadedSpeechClip_ProjVQ,
    KeywordCascadedSpeechClip_ProjVQ_Cosine,
    VQCascadedSpeechClip,
)

def readArgs(pathArgs):
    print(f"Loading args from {pathArgs}")
    with open(pathArgs, 'r') as file:
        args = argparse.Namespace(**json.load(file))
    return args

def writeArgs(pathArgs, args):
    print(f"Writing args to {pathArgs}")
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export BERT features from quantized units of audio files.')
    parser.add_argument('filelisttxt', type=str,
                        help='Path to the quantized units. Each line of the input file must be'
                        'of the form file_name[tab]pseudo_units (ex. hat  1,1,2,3,4,4)')
    parser.add_argument("model_type", type=str, help="CascadedClip model type, KeywordCascadedSpeechClip, KeywordCascadedSpeechClip_ProjVQ,"
                        "KeywordCascadedSpeechClip_ProjVQ_Cosine, VQCascadedSpeechClip")
    parser.add_argument('pathmodelCheckpoint', type=str,
                    help='Path to the ckpt.')
    parser.add_argument('pathOutputDir', type=str,
                        help='Path to the output directory.')
    parser.add_argument('--dict', type=str,
                       help='Path to the dictionary file (dict.txt) used to train the BERT model'
                       '(if not speficied, look for dict.txt in the model directory)')
    parser.add_argument('--hidden_level', type=int, default=-1,
                          help="Hidden layer of BERT to extract features from (default: -1, last layer).")
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    parser.add_argument('--cpu', action='store_true',
                        help="Run on a cpu machine.")
    return parser.parse_args(argv)

def main(argv):
    # Args parser
    args = parseArgs(argv)

    # Load input file
    print("")
    print(f"Reading input file from {args.filelisttxt}")
    wav_names = []
    wav_files = []
    with open(args.filelisttxt, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                root_path = line.strip().split("\t")[0]
                continue
            else:
                wav_name = line.strip().split("\t")[0]
                wav_names.append(wav_name)
                wav_files.append(os.path.join(root_path, wav_name))

    print(f"Found {len(wav_files)} Inputs!")

    # Verify the output directory
    if os.path.exists(args.pathOutputDir):
        existing_files = set([os.path.splitext(os.path.basename(x))[0]
                            for x in os.listdir(args.pathOutputDir) if x[-4:]==".npy"])
        wav_names = [s for s in wav_names if os.path.splitext(os.path.basename(s[1]))[0] not in existing_files]
        print(f"Found existing output directory at {args.pathOutputDir}, continue to build features of {len(wav_names)} audio files left!")
    else:
        print("")
        print(f"Creating the output directory at {args.pathOutputDir}")
        Path(args.pathOutputDir).mkdir(parents=True, exist_ok=True)
    # writeArgs(os.path.join(args.pathOutputDir, "_info_args.json"), args)

    # Debug mode
    if args.debug:
        nsamples=20
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        # shuffle(wav_names)
        wav_names = wav_names[:nsamples]
        wav_files = wav_files[:nsamples]


    print(f"Loading CascadedClip model from {args.pathmodelCheckpoint}...")
    

    if args.model_type == "KeywordCascadedSpeechClip":
        model = KeywordCascadedSpeechClip.load_from_checkpoint(args.pathmodelCheckpoint)
    elif args.model_type == "KeywordCascadedSpeechClip_ProjVQ":
        model = KeywordCascadedSpeechClip_ProjVQ.load_from_checkpoint(args.pathmodelCheckpoint)
    elif args.model_type == "KeywordCascadedSpeechClip_ProjVQ_Cosine":
        model = KeywordCascadedSpeechClip_ProjVQ_Cosine.load_from_checkpoint(args.pathmodelCheckpoint)
    elif args.model_type == "VQCascadedSpeechClip":
        model = VQCascadedSpeechClip.load_from_checkpoint(args.pathmodelCheckpoint)
    else:
        print(f"Not supported model type {args.model_type}")
        
    model.eval()  # disable dropout (or leave in train mode to finetune)
    # if not args.cpu:
    #     model.cuda()
    print("Model loaded !")

    # Define BERT_feature_function
    # def BERT_feature_function(input_sequence, n_hidden=-1):
    #     sentence_tokens = roberta.task.source_dictionary.encode_line(
    #                         "<s> " + input_sequence,
    #                         append_eos=True,
    #                         add_if_not_exist=False).type(torch.LongTensor)
    #     if not args.cpu:
    #         sentence_tokens = sentence_tokens.cuda()

    #     with torch.no_grad():
    #         outputs = roberta.extract_features(sentence_tokens, return_all_hiddens=True)

    #     return outputs[n_hidden].squeeze(0).float().cpu().numpy()

    def CascadedClip_feature_function(input_sequence):
        outputs = model.feature_extractor_zerospeech(input_sequence, using_keywords=True)
        outputs = outputs.squeeze(0).float().detach().cpu().numpy()
        assert outputs.ndim == 2
        return outputs
        

    # Building features
    print("")
    print(f"Building Cascadedclip features and saving outputs to {args.pathOutputDir}...")
    bar = progressbar.ProgressBar(maxval=len(wav_names))
    bar.start()
    start_time = time()
    for index, (name_seq, wav_path) in enumerate(zip(wav_names, wav_files)):
        bar.update(index)

        # Computing features
        wav = torch.FloatTensor(librosa.load(wav_path, sr=16_000)[0])
        audo_features = CascadedClip_feature_function([wav])

        # Save the outputs
        file_name = os.path.splitext(name_seq)[0] + ".txt"
        file_out = os.path.join(args.pathOutputDir, file_name)
        np.savetxt(file_out, audo_features)
        
    bar.finish()
    print(f"...done {len(wav_names)} files in {time()-start_time} seconds.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
