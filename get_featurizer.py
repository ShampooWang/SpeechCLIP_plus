import argparse
from unittest import result
import torch
import os
from os.path import join
from tkinter import font
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib as mpl
from matplotlib import rcParams
from brokenaxes import brokenaxes
import numpy as np
STYLE = 'default'
TASK = "SF"
# STYLE = 'seaborn-whitegrid'
# STYLE = 'seaborn-notebook'


# layer_labels = [
#     'feat',
#     'hid 1',
#     'hid 2',
#     'hubert 4',
#     'hubert 8',
#     'hubert 12'
# ]
# results = {
#     'PR': [0.00967, 0.00000, 0.11330, 0.08935, 0.77624, 0.01144],
#     'KS': [0.26162, 0.07494, 0.26380, 0.04024, 0.06108, 0.29832],
#     'IC': [0.12775, 0.01261, 0.41610, 0.00248, 0.01163, 0.42941],
#     'SID': [0.04363, 0.00000, 0.95637, 0.00000, 0.00000, 0.00000],
#     'ER': [0.28974, 0.15477, 0.25882, 0.07287, 0.07064, 0.15316],
#     'ASR': [0.12278, 0.00066, 0.62146, 0.00280, 0.22822, 0.02408],
#     'SF': [0.20298, 0.00256, 0.64659, 0.00317, 0.11466, 0.03004],
#     'ASV': [0.19654, 0.14184, 0.43778, 0.12009, 0.03768, 0.06607],
#     'SD': [0.30905, 0.15379, 0.18404, 0.10397, 0.09100, 0.15815]
# }

# WIDTH = 0.08
# colors = [
#     'orangered',
#     'orange',
#     'gold',
#     'lawngreen',
#     'turquoise',
#     'dodgerblue',
#     'mediumblue',
#     'blueviolet',
#     'deeppink'
# ]


# layer_labels = ['CNN'] + [str(i) for i in range(1, 25)] + ['Trm']

# l2_norm = {
#     'HuBERT Large': [2197.7473, 2234.3003, 2362.8691, 2541.5337, 2758.8035, 2895.336, 3085.5566, 3345.6729, 3392.6938, 3556.2134, 3780.0947, 3931.616, 4174.961, 4829.773, 5096.7886, 5993.7407, 6484.0405, 7181.5176, 7634.812, 8288.421, 8698.396, 10159.086, 15395.428, 18987.232, 148.55734, 0],
#     'Parallel Large': [584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.90027],
#     'Cascaded Large': [584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.90216, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.9022, 584.90027]
# }

# results = {
#     'HuBERT Large': [0.005837019067, 0.003234192962, 0.00133770192, 0.0007252070354, 0.0003841939324, 0.0001910970168, 9.30E-05, 
#     4.79E-05, 2.22E-05, 2.34E-05, 4.47E-05, 0.0002988450578, 0.002623895882, 0.03317164257, 0.001382715767, 0.003747349372, 0.01984396204, 0.02769423835, 0.009760960937,	
#     0.006748188753,	0.01381463557, 0.01845585927, 0.0108319968, 0.003454373917, 0.8362308741, 0],
#     'Parallel Large': [0.009173285216, 0.006109887734, 0.003473107237, 0.00235948991, 0.00164447946, 0.001069140621, 0.0006824959419, 
#     0.0004698040721, 0.000284597656, 0.0003303326084, 0.0006291967002, 0.002828848083, 0.01382500771, 0.1056806743, 0.01078903861, 
#     0.02284206264, 0.09735933691, 0.1252339184, 0.0586524196, 0.04951342195, 0.08773271739, 0.1472078264, 0.1248727664, 0.07835109532, 0.0300396625, 0.0188453868],
#     'Cascaded Large': [0.009049075656, 0.005917599425, 0.003491268959, 0.00239532534, 0.001809992827, 0.001216923469, 0.0008352377918, 0.0005852266913, 0.0003976488952, 
#     0.0004537511559, 0.000869917043, 0.003305038437, 0.01507619116, 0.09633856267, 0.01223131362, 0.02392658405, 0.09566830099, 0.1182734445, 0.0595414713, 0.04999144748,	
#     0.08994242549, 0.1504306644, 0.125370875, 0.07781547308, 0.03132241964, 0.0237438418]
# }

WIDTH = 0.2
# colors = [
#     # 'orangered',
#     'tomato',
#     'gold',
#     # 'lightgreen',
#     # 'turquoise',
#     # 'dodgerblue',
#     'cornflowerblue',
#     # 'blueviolet',
#     'mediumorchid',
#     'pink'
# ]
colors = {
    "SF": 'turquoise',
    "KS": '#E48B90', # 'turquoise', '#E48B90', '#FCD460'
}
# hatches = ['///', '.', '--', '++']


def plot_weights(out_dir, results, layer_labels, l2_norm=None):
    # figure(figsize=(4, 3))
    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    x = np.arange(len(layer_labels))
    width = WIDTH
    for i, (key, val) in enumerate(results.items()):
        if l2_norm is not None:
            val = [weight * norm for weight, norm in zip(results[key], l2_norm[key])]
        else:
            val = [weight for weight in results[key]]
        # print(key, val)
        ax.bar(x + (i - len(results) / 2 + 0.5) * width,
               np.array(val), width, label=key,
               color=[colors[key]] * len(val))

    ax.set_xlabel('Layer')
    ax.set_xticks(x)
    # ax.set_xlim([-0.5, 25.5])
    ax.set_xticklabels(layer_labels)
    ax.set_ylim(0, 1.0)

    ax.legend(framealpha=1.0, frameon=True)
    # ax.legend(loc=9, ncol=5)
    fig.tight_layout()

    if os.path.exists(out_dir):
        path = join(out_dir, 'weights.png')
        plt.savefig(path, bbox_inches='tight')
        print(f'Saved results to {path}')

    plt.show()

SCALE_NUM = 5.0
SCALE_INTERVAL = 0.04
def plot_weights_split(out_dir):
    # figure(figsize=(4, 3))
    # fig, ax = plt.subplots(figsize=(5.5, 2.8))
    # fig = plt.figure(figsize=(5.5, 2.5))
    fig = plt.figure(figsize=(4, 1.45))
    # bax = brokenaxes(
    #     ylims=((0., 0.25), (0.85, 1.0)),
    #     hspace=0.1, tilt=30,
    #     despine=False)
    bax = brokenaxes(despine=False)

    x = np.arange(len(results['HuBERT Large']))
    weight_labels = np.arange(0, SCALE_NUM * SCALE_INTERVAL, SCALE_INTERVAL)
    width = WIDTH
    for i, (key, val) in enumerate(results.items()):
        val = [weight * norm for weight, norm in zip(results[key], l2_norm[key])]
        val = [i / sum(val) for i in val]
        # print(key, val)
        bax.bar(x + (i - len(results) / 2 + 0.5) * width,
                np.array(val), width, label=key,
                color=[colors[key]] * len(val))
        # hatch=hatches[i])

    # ===
    # Set additional diags
    size = bax.fig.get_size_inches()
    ylen = bax.d * np.sin(bax.tilt * np.pi / 180) * size[0] / size[1]
    xlen = bax.d * np.cos(bax.tilt * np.pi / 180)
    d_kwargs = dict(
        transform=bax.fig.transFigure,
        color=bax.diag_color,
        clip_on=False,
        lw=rcParams["axes.linewidth"],
    )

    # print(rcParams)
    # xpos = 2 / 15 + width * 1.5
    # xpos = 0.239
    # for ax in bax.axs:
    #     bounds = ax.get_position().bounds
    #     if ax.get_subplotspec().is_last_row():
    #         ypos = bounds[1]
    #         if not ax.get_subplotspec().is_last_col():
    #             bax.draw_diag(ax, xpos, xlen, ypos, ylen, **d_kwargs)
    #         if not ax.get_subplotspec().is_first_col():
    #             bax.draw_diag(ax, xpos, xlen, ypos, ylen, **d_kwargs)

    #     if ax.get_subplotspec().is_first_col():
    #         # xpos = bounds[0]
    #         if not ax.get_subplotspec().is_first_row():
    #             ypos = bounds[1] + bounds[3]
    #             bax.draw_diag(ax, xpos, xlen, ypos, ylen, **d_kwargs)
    #         if not ax.get_subplotspec().is_last_row():
    #             ypos = bounds[1]
    #             bax.draw_diag(ax, xpos, xlen, ypos, ylen, **d_kwargs)
    # ===

    # bax.axs[1].set_xlabel('Hidden Layer')
    # bax.axs[1].set_xticks(x)
    # bax.axs[1].set_xticklabels(layer_labels)

    bax.axs[0].set_xlabel('Hidden Layer', fontsize=6)
    bax.axs[0].set_xticks(x)
    bax.axs[0].set_xlim([-0.5, 25.5])
    bax.axs[0].set_xticklabels(layer_labels, fontsize=5, rotation=0)
    bax.axs[0].set_yticks(weight_labels)
    bax.axs[0].set_yticklabels(weight_labels, fontsize=5, rotation=0)
    # bax.legend(framealpha=1.0, frameon=True)
    bax.legend(framealpha=1.0, fontsize=5, frameon=True, loc=2, ncol=1)

    for axis in ['top','bottom','left','right']:
        bax.axs[0].spines[axis].set_linewidth(0.1)
    bax.axs[0].tick_params(length=2,width=0.1)

    if os.path.exists(out_dir):
        path = join(out_dir, 'asr_large_coco_weights.pdf')
        plt.savefig(path, bbox_inches='tight')
        print(f'Saved results to {path}')

    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    args = parser.parse_args()

    with torch.no_grad():
        ckpt = torch.load(args.ckpt)
        weights = ckpt["Featurizer"]["weights"]
        weights = torch.softmax(weights, 0).cpu().tolist()
        for w in weights:
            print(w)

    mpl.style.use(STYLE)
    results = {}
    results[TASK] = weights
    # layer_labels = [f"b+_{i}" for i in range(13)] + [f"c+_{i}" for i in range(13)]
    layer_labels = [str(i) for i in range(26)]
    # print(layer_labels)
    plot_weights('/mnt/md0/user_jeff/SUPERB/wavlm/SF/Flickr_c+_small_ensemble', results=results, layer_labels=layer_labels)