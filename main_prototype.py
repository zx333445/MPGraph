from __future__ import print_function
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.wsi_prototype import WSIProtoDataset
from utils.file_utils import save_pkl
from utils.proto_utils import cluster


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Prototype')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=8)

# Cluster args 
parser.add_argument('--n_proto', type=int, default=8, help='Number of prototypes')
parser.add_argument('--n_proto_patches', type=int, default=250000, help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
parser.add_argument('--n_init', type=int, default=5, help='Number of different KMeans initialization (for FAISS)')
parser.add_argument('--n_iter', type=int, default=50, help='Number of iterations for Kmeans clustering')
parser.add_argument('--in_dim', type=int, default=1024, help='Dimension of the input features')
parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='kmeans')

# Data args
parser.add_argument('--data_dir', type=str, default='/home/stat-zx/6.TCGA-CRC/unifeats', help='manually specify the data dir')
parser.add_argument('--csv_path', type=str, default='/home/stat-zx/CRsurv/csvfiles/train.csv', help='manually specify the csv path')
parser.add_argument('--save_dir', type=str, default='./data', help='directory for saving prototypes')
args = parser.parse_args()


def seed_torch(seed=7):
    import random
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):  
    seed_torch(args.seed)
    dataset = WSIProtoDataset(args.data_dir, args.csv_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print('Data loaded')
    _, weights = cluster(loader,
                            n_proto=args.n_proto,
                            n_iter=args.n_iter,
                            n_init=args.n_init,
                            feature_dim=args.in_dim,
                            mode=args.mode,
                            n_proto_patches=args.n_proto_patches,
                            use_cuda=True if torch.cuda.is_available() else False)
    
    save_name = f"{args.n_proto}proto_{args.mode}_num_{args.n_proto_patches:.1e}.pkl"
    save_fpath = os.path.join(args.save_dir, save_name)
    save_pkl(save_fpath, {'prototypes': weights})
    print("Saved to:")
    print(save_fpath)


if __name__ == "__main__":

    main(args)