# README

# SpeechCLIP+

### Reminder: this repository also supports the original SpeechCLIP usage (e.g. loading checkpoint, training and testing)


<div style="text-align: center;">
    <div style="display: inline-block; width: 25%; text-align: center;">
        <img alt="Hybrid Model" src="hybrid.png" style="max-width: 45%;"/>
        &nbsp;
        &nbsp;
        <img alt="Hybrid+ Model" src="hybrid+.png" style="max-width: 46.5%;"/>
        <br/>
        <span>Left: Hybrid SpeechCLIP. Right: Cascaded and Hybrid SpeechCLIP+</span>
    </div>
    <br/>
    <br/>
    <div style="text-align: center;">
        <a href=""><img alt="LICENSE" src="https://img.shields.io/github/license/ShampooWang/SpeechCLIP_plus"/></a>
        <a href=""><img alt="STAR" src="https://img.shields.io/github/stars/ShampooWang/SpeechCLIP_plus"/></a>
        <a href="https://github.com/ShampooWang/SpeechCLIP_plus/issues"><img alt="ISSUE" src="https://img.shields.io/github/issues/ShampooWang/SpeechCLIP_plus"/></a>
        <a href="https://github.com/ShampooWang/SpeechCLIP_plus/pulls"><img alt="PR" src="https://img.shields.io/github/issues-pr/ShampooWang/SpeechCLIP_plus"/></a>
    </div>
</div>



Links: [arXiv](https://arxiv.org/abs/2402.06959)

## Code Contributors

Hsuan-Fu Wang, Yi-Jen Shih, Heng-Jui Chang

## Prequisite

### Install packages

```bash
pip install -r requirements.txt
```

### Data Preparation

See [Details](data/README.md)

### Download Pretrained
Checkpoints

```bash
bash download_ckpts.sh
```

You could see `Done downloading all checkpoints` after
the script is executed

> Notice that it reuqires 2 GPUs for training base models and 4 GPUs
for large models
> 

## Usage

Remember to check the `dataset_root` ### Train

Example: train Cascaded SpeechCLIP+ base:

```bash
bash egs/speechCLIP+/model_base/cascaded+/train.sh
```

### Inference

Example: test Parallel SpeechCLIP base: (Using pretrained
checkpoint)

```bash
bash egs/speechCLIP+/model_base/cascaded+/test.sh
```

> For more settings, please see the folders in ./egs/.
> 

### Getting embeddings from
SpeechCLIP or SpeechCLIP+

See [example.py](example.py)

## Citation

```
@article{wang2024speechclip+,
  title={SpeechCLIP+: Self-supervised multi-task representation learning for speech via CLIP and speech-image data},
  author={Wang, Hsuan-Fu and Shih, Yi-Jen and Chang, Heng-Jui and Berry, Layne and Peng, Puyuan and Lee, Hung-yi and Wang, Hsin-Min and Harwath, David},
  journal={arXiv preprint arXiv:2402.06959},
  year={2024}
}
```

## TBD

- Release the code of keyword evaluation (urgent!).
- Clean the comments on config. files.

## Contribute

Please run autoformatter before opening PR! Autoformat
`./dev-support/`