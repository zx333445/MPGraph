"""
Hard-clustering-based aggregation

Ref:
    Vu, Quoc Dang, et al. "Handcrafted Histological Transformer (H2T): Unsupervised representation of whole slide images." Medical image analysis 85 (2023): 102743.
"""

import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

from utils.file_utils import save_pkl, load_pkl


class H2T(nn.Module):
    """
    WSI is represented as a prototype-count vector
    """
    def __init__(self, config, mode):
        super().__init__()

        assert config.load_proto, "Prototypes must be loaded!"
        assert os.path.exists(config.proto_path), "Path {} doesn't exist!".format(config.proto_path)

        self.config = config
        self.mode = mode
        proto_path = config.proto_path

        if proto_path.endswith('pkl'):
            weights = load_pkl(proto_path)['prototypes'].squeeze()
        elif proto_path.endswith('npy'):
            weights = np.load(proto_path)

        self.n_proto = config.n_proto
        self.prototypes = torch.from_numpy(weights).float()
        self.prototypes = self.prototypes / torch.norm(self.prototypes, dim=1).unsqueeze(1)

        emb_dim = config.in_dim

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        self.prototypes = self.prototypes.to(x.device)

        x = x / torch.norm(x, dim=-1).unsqueeze(2)
        dist = torch.cdist(self.prototypes, x, p=2) # (1 x n_proto x n_instances)
        c_identity = torch.argmin(dist.squeeze(), dim=0) # (n_instances)

        feats = []
        for idx in range(self.n_proto):
            indices = torch.nonzero(c_identity == idx)
            if len(indices) != 0:
                feat = torch.mean(x[:, indices, :], dim=1)
            else:
                feat = torch.zeros((1,1,x.shape[-1])).to(x.device)
            feats.append(feat)
        out = torch.cat(feats, dim=1)
        out = out.reshape(x.shape[0], self.n_proto, -1)
        return {'repr': out}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    def predict_emb(self, dataset, use_cuda=True, permute=False):
        """
        Create prototype-based slide representation

        Returns
        - X (torch.Tensor): (n_data x output_set_dim)
        - y (torch.Tensor): (n_data)
        """

        X = []
        y = None
        for i in tqdm(range(len(dataset))):
            batch = dataset.__getitem__(i)
            data = batch['img'].unsqueeze(dim=0)
            if use_cuda:
                data = data.cuda()
            
            with torch.no_grad():
                out = self.representation(data)
                out = out['repr'].data.detach().cpu()

            X.append(out)

        X = torch.cat(X)

        return X, y