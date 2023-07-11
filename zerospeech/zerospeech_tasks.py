from math import ceil
import os
import glob
import shutil
import librosa
import logging
import numpy as np
import torch
import tqdm
import random
from torch.utils.data import Dataset, DataLoader
logger = logging.getLogger(__name__)

__all__ = ["Task_semantic", "Task_phonetic"]

class AudioDataset(Dataset):
    def __init__(self,wav_paths,sr):
        super().__init__()
        self.wav_paths = wav_paths
        self.sr = sr
    def __len__(self):
        return len(self.wav_paths)
    def __getitem__(self, index):
        return torch.FloatTensor(librosa.load(self.wav_paths[index], sr=self.sr)[0]), self.wav_paths[index]


class Task_base():
    TASKS_NAME = ["lexical", "phonetic",  "semantic",  "syntactic"]
    def __init__(self,my_model,output_result_dir,inference_bsz=64,sample_rate=16_000,run_dev=False,run_test=False,**kwarg):
        self.run_dev = run_dev
        self.run_test = run_test
        self.sample_rate = sample_rate
        self.inference_bsz = inference_bsz
        self.my_model = my_model
        self.output_result_dir = output_result_dir
        # if os.path.exists(self.output_result_dir):
        #     print("output_result_dir({}) exists!".format(self.output_result_dir))
        #     exit(1)
        # else:
        os.makedirs(self.output_result_dir,exist_ok=True)

        for _tn in self.TASKS_NAME:
            os.makedirs(os.path.join(self.output_result_dir,_tn),exist_ok=True)

        # create meta yaml

        shutil.copy("./meta.yaml",os.path.join(self.output_result_dir,"meta.yaml"))

class Task_semantic(Task_base):
    DATA_SOURCE_NAME = ["librispeech","synthetic"]
    def __init__(
        self,
        model_cls_name,
        model_ckpt,
        task_input_dir,
        task_name,
        **kwarg):
        super().__init__(**kwarg)
        self.model_cls_name = model_cls_name
        self.model_ckpt = model_ckpt
        self.task_input_dir = task_input_dir
        self.task_name = task_name
        

        self.task_root_dir = os.path.join(self.output_result_dir,"semantic")

        assert os.path.exists(self.task_root_dir)

        self.audio_wav_paths = {}
        if self.run_dev:
            dev_libri_path_txt = os.path.join(self.task_input_dir,"dev_librispeech.txt")
            dev_synthetic_path_txt = os.path.join(self.task_input_dir,"dev_synthetic.txt")
            os.makedirs(os.path.join(self.task_root_dir,"dev"),exist_ok=True)
            os.makedirs(os.path.join(self.task_root_dir,"dev","librispeech"),exist_ok=True)
            os.makedirs(os.path.join(self.task_root_dir,"dev","synthetic"),exist_ok=True)
            
            with open(dev_libri_path_txt,"r") as f:
                _data = [ os.path.join(self.task_input_dir,"dev","librispeech",x.strip())    for x in f.readlines() if x.strip().endswith(".wav")]
                self.audio_wav_paths["dev_librispeech"] = _data

            with open(dev_synthetic_path_txt,"r") as f:
                _data = [os.path.join(self.task_input_dir,"dev","synthetic", x.strip()) for x in f.readlines() if x.strip().endswith(".wav")]
                self.audio_wav_paths["dev_synthetic"] = _data
        if self.run_test:
            test_libri_path_txt = os.path.join(self.task_input_dir,"test_librispeech.txt")
            test_synthetic_path_txt = os.path.join(self.task_input_dir,"test_synthetic.txt")
            os.makedirs(os.path.join(self.task_root_dir,"test"),exist_ok=True)
            os.makedirs(os.path.join(self.task_root_dir,"test","librispeech"),exist_ok=True)
            os.makedirs(os.path.join(self.task_root_dir,"test","synthetic"),exist_ok=True)
            
            with open(test_libri_path_txt,"r") as f:
                _data = [ os.path.join(self.task_input_dir,"test","librispeech",x.strip()) for x in f.readlines() if x.strip().endswith(".wav")]
                self.audio_wav_paths["test_librispeech"] = _data

            with open(test_synthetic_path_txt,"r") as f:
                _data = [os.path.join(self.task_input_dir,"test","synthetic", x.strip()) for x in f.readlines() if x.strip().endswith(".wav")]
                self.audio_wav_paths["test_synthetic"] = _data
        

    def run(self, feat_select_idx: int):
        self.my_model.eval()
        self.my_model.cuda()
        if self.run_dev:
            print(f"Inferencing zerospeech {self.task_name} dev")
            for _split in self.DATA_SOURCE_NAME:
                print("Datasource: {}, total:{}, bsz:{} ".format(
                    _split,
                    len(self.audio_wav_paths["{}_{}".format("dev",_split)]),
                    self.inference_bsz,
                    ceil(len(self.audio_wav_paths["{}_{}".format("dev",_split)]) / self.inference_bsz)
                ))
                # _dataset = AudioDataset(
                #     wav_paths=self.audio_wav_paths["{}_{}".format("dev",_split)],
                #     sr=self.sample_rate
                # )
                # dev_dataloader = DataLoader(
                #     dataset=_dataset,
                #     batch_size=self.inference_bsz,
                #     shuffle=False,
                #     num_workers=8,
                # )

                for i in tqdm.tqdm(range(0,len(self.audio_wav_paths["{}_{}".format("dev",_split)])+self.inference_bsz,self.inference_bsz)):
                # for _data,_wavpaths in tqdm.tqdm(dev_dataloader):
                    _wavpaths = self.audio_wav_paths["{}_{}".format("dev",_split)][i:i+self.inference_bsz]
                    # print(_wavpaths)[0]
                    # exit()
                    if len(_wavpaths) == 0 : continue
                    _data = []
                    for _w in _wavpaths:
                        _data.append(
                            torch.FloatTensor(librosa.load(_w, sr=self.sample_rate)[0]).cuda()
                        )
                    # _data = [x.cuda() for x in _data]
                    
                    with torch.no_grad():
                        embeddings = self.my_model.feature_extractor_zerospeech(wav=_data, feat_select_idx=feat_select_idx)
                        if isinstance(embeddings, torch.Tensor):
                            embeddings = embeddings.cpu().float().numpy()
                    
                    for _embs, _wavpath in zip(embeddings,_wavpaths):
                        assert _embs.shape[0] > 1, _embs.shape
                        _embs = np.atleast_2d(_embs)
                        txt_path = os.path.join(self.task_root_dir,"dev",_split,os.path.basename(_wavpath).replace(".wav",".txt"))
                        np.savetxt(txt_path,_embs)

            print(f"Done inferencing zerospeech {self.task_name} dev")

        if self.run_test:
            print(f"Inferencing zerospeech {self.task_name} test")
            for _split in self.DATA_SOURCE_NAME:
                print("Datasource: {}, total:{}, bsz:{} ".format(
                    _split,
                    len(self.audio_wav_paths["{}_{}".format("test",_split)]),
                    self.inference_bsz,
                    ceil(len(self.audio_wav_paths["{}_{}".format("test",_split)]) / self.inference_bsz)
                ))
                # _dataset = AudioDataset(
                #     wav_paths=self.audio_wav_paths["{}_{}".format("dev",_split)],
                #     sr=self.sample_rate
                # )
                # dev_dataloader = DataLoader(
                #     dataset=_dataset,
                #     batch_size=self.inference_bsz,
                #     shuffle=False,
                #     num_workers=8,
                # )

                for i in tqdm.tqdm(range(0,len(self.audio_wav_paths["{}_{}".format("test",_split)])+self.inference_bsz,self.inference_bsz)):
                # for _data,_wavpaths in tqdm.tqdm(dev_dataloader):
                    _wavpaths = self.audio_wav_paths["{}_{}".format("test",_split)][i:i+self.inference_bsz]
                    # print(_wavpaths)[0]
                    # exit()
                    if len(_wavpaths) == 0 : continue
                    _data = []
                    for _w in _wavpaths:
                        _data.append(
                            torch.FloatTensor(librosa.load(_w, sr=self.sample_rate)[0]).cuda()
                        )
                    # _data = [x.cuda() for x in _data]
                    with torch.no_grad():
                        embeddings = self.my_model.feature_extractor_zerospeech(wav=_data, feat_select_idx=feat_select_idx)
                        if isinstance(embeddings, torch.Tensor):
                            embeddings = embeddings.cpu().float().numpy()

                    # print(embeddings.shape)
                    for _embs, _wavpath in zip(embeddings,_wavpaths):
                        assert _embs.shape[0] > 1, _embs.shape
                        _embs = np.atleast_2d(_embs)
                        txt_path = os.path.join(self.task_root_dir,"test",_split,os.path.basename(_wavpath).replace(".wav",".txt"))
                        np.savetxt(txt_path,_embs)

            print(f"Done inferencing zerospeech {self.task_name} test")

class Task_phonetic(Task_base):
    DATA_SOURCE_NAME = ["dev-clean","dev-other", 'test_clean', 'test_other']
    def __init__(
        self,
        model_cls_name,
        model_ckpt,
        task_input_dir,
        task_name,
        **kwarg):
        super().__init__(**kwarg)
        self.model_cls_name = model_cls_name
        self.model_ckpt = model_ckpt
        self.task_input_dir = task_input_dir
        self.task_name = task_name
        

        self.task_root_dir = os.path.join(self.output_result_dir,"phonetic")

        assert os.path.exists(self.task_root_dir)

        self.audio_wav_paths = {}
        if self.run_dev:
            os.makedirs(os.path.join(self.task_root_dir,"dev-clean"), exist_ok=True)
            os.makedirs(os.path.join(self.task_root_dir,"dev-other"), exist_ok=True)
            self.audio_wav_paths["dev-clean"] = [ fn for fn in glob.glob(os.path.join(self.task_input_dir,"dev-clean", "*.wav")) ]
            self.audio_wav_paths["dev-other"] = [ fn for fn in glob.glob(os.path.join(self.task_input_dir,"dev-other", "*.wav")) ]

        if self.run_test:
            os.makedirs(os.path.join(self.task_root_dir,"test-clean"), exist_ok=True)
            os.makedirs(os.path.join(self.task_root_dir,"test-other"), exist_ok=True)
            self.audio_wav_paths["test-clean"] = [ fn for fn in glob.glob(os.path.join(self.task_input_dir,"test-clean", "*.wav")) ]
            self.audio_wav_paths["test-other"] = [ fn for fn in glob.glob(os.path.join(self.task_input_dir,"test-other", "*.wav")) ]
        

    def run(self, feat_select_idx: int):
        self.my_model.eval()
        self.my_model.cuda()
        for _split in self.audio_wav_paths.keys():
            data_size = len(self.audio_wav_paths[_split])
            print(f"Inferencing zerospeech {self.task_name} {_split}")
            print(f"total:{data_size}, bsz:{self.inference_bsz} ")

            for i in tqdm.tqdm(range(0, data_size + self.inference_bsz, self.inference_bsz)):
                _wavpaths = self.audio_wav_paths[_split][i:i+self.inference_bsz]
                # _wavpaths = [ os.path.join(self.task_input_dir, _split, _wn + ".wav") for _wn in _wavnames]

                if len(_wavpaths) == 0 : continue

                _data = []
                for _w in _wavpaths:
                    _data.append(
                        torch.FloatTensor(librosa.load(_w, sr=self.sample_rate)[0]).cuda()
                    )

                with torch.no_grad():
                    embeddings = self.my_model.feature_extractor_zerospeech(wav=_data, feat_select_idx=feat_select_idx)
                    if isinstance(embeddings, torch.Tensor):
                        embeddings = embeddings.cpu().float().numpy()

                for _embs, _wavpath in zip(embeddings, _wavpaths):
                    assert _embs.shape[0] > 1, _embs.shape
                    _embs = np.atleast_2d(_embs)
                    txt_path = os.path.join(self.task_root_dir, _split, os.path.basename(_wavpath).replace(".wav",".txt"))
                    np.savetxt(txt_path,_embs)

            print(f"Done inferencing zerospeech {self.task_name} {_split}")