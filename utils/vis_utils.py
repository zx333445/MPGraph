import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager as fm, rcParams
from PIL import Image, ImageOps

import sys
sys.path.append('../')
from models.panther.model_panther import PANTHER


colors1 = [
        '#696969','#556b2f','#a0522d','#483d8b', 
        '#008000','#008b8b','#000080','#7f007f',
        '#8fbc8f','#b03060','#ff0000','#ffa500',
        '#00ff00','#8a2be2','#00ff7f','#FFFF54', 
        '#00ffff','#00bfff','#f4a460','#adff2f',
        '#da70d6','#b0c4de','#ff00ff','#1e90ff',
        '#f0e68c','#0000ff','#dc143c','#90ee90',
        '#ff1493','#7b68ee','#ffefd5','#ffb6c1']

colors2 = ["#90ee90", "#008b8b", "#f0e68c", "#00bfff", "#7b68ee", "#a0522d", "#b03060", "#f4a460"]


def get_mixture_plot(mixtures):
    colors = colors2

    cmap = {f'c{k}':v for k,v in enumerate(colors[:len(mixtures)])}
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.bottom'] = True
    fig = plt.figure(figsize=(6,3), dpi=300)

    prop = fm.FontProperties(fname="./Arial.ttf")
    mpl.rcParams['axes.linewidth'] = 1.3
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    mixtures = pd.DataFrame(mixtures, index=cmap.keys()).T # type: ignore
    ax = sns.barplot(mixtures, palette=cmap)
    plt.axis('on')
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)
    ax.set_xlabel('Cluster', fontproperties=prop, fontsize=12)
    ax.set_ylabel('Proportion / Mixture', fontproperties=prop, fontsize=12)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontproperties = prop, fontsize=12) # type: ignore
    ax.set_ylim([0, 0.55]) # type: ignore
    plt.close()
    return ax.get_figure()

def hex_to_rgb_mpl_255(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x*255) for x in rgb])

def get_default_cmap(n=32):
    colors = colors2
    
    colors = colors[:n]
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict


def get_panther_encoder(in_dim, p, proto_path, iter):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='PANTHER')
    parser.add_argument('--proto_model_type', type=str, default='PANTHER')
    parser.add_argument('--model_config', type=str, default='PANTHER_default')
    parser.add_argument('--in_dim', type=int, default=in_dim)
    parser.add_argument('--n_proto', type=int, default=p)
    parser.add_argument('--em_iter', type=int, default=iter)
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--out_type', type=str, default='allcat')
    parser.add_argument('--load_proto', type=int, default=1)
    parser.add_argument('--ot_eps', type=int, default=1)
    args = parser.parse_known_args()[0]
    args.fix_proto = 1
    args.proto_path = proto_path

    model = PANTHER(args, mode = 'emeb')
    return model

def visualize_categorical_heatmap(
        wsi,
        coords, 
        labels, 
        label2color_dict,
        vis_level=None,
        patch_size=(256, 256),
        canvas_color=(255, 255, 255),
        alpha=0.4,
        verbose=True,
    ):

    # Scaling from 0 to desired level
    downsample = int(wsi.level_downsamples[vis_level])
    scale = [1/downsample, 1/downsample]

    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)

    top_left = (0, 0)
    bot_right = wsi.level_dimensions[0]
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    w, h = region_size

    patch_size_orig = patch_size
    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    if verbose:
        print('\nCreating heatmap for: ')
        print('Top Left: ', top_left, 'Bottom Right: ', bot_right)
        print('Width: {}, Height: {}'.format(w, h))
        print(f'Original Patch Size / Scaled Patch Size: {patch_size_orig} / {patch_size}')
    
    vis_level = wsi.get_best_level_for_downsample(downsample)
    img = wsi.read_region(top_left, vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    if img.size != region_size:
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    
    if verbose:
        print('vis_level: ', vis_level)
        print('downsample: ', downsample)
        print('region_size: ', region_size)
        print('total of {} patches'.format(len(coords)))
    
    for idx in tqdm(range(len(coords))):
        coord = coords[idx]
        color = label2color_dict[labels[idx][0]]
        img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
        color_block = (np.ones((img_block.shape[0], img_block.shape[1], 3)) * color).astype(np.uint8)
        blended_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0)
        blended_block = np.array(ImageOps.expand(Image.fromarray(blended_block), border=1, fill=(50,50,50)).resize((img_block.shape[1], img_block.shape[0]))) # type: ignore
        img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = blended_block

    img = Image.fromarray(img)
    return img