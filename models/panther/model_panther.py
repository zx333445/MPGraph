# Model initiation for PANTHER
import torch
import torch.nn as nn

from tqdm import tqdm
from models.panther.base_layers import PANTHERBase
from utils.proto_utils import check_prototypes


class PANTHER(nn.Module):
    """
    Wrapper for PANTHER model
    """
    def __init__(self, config, mode):
        super(PANTHER, self).__init__()

        self.config = config
        emb_dim = config.in_dim

        self.emb_dim = emb_dim
        self.n_proto = config.n_proto
        self.load_proto = config.load_proto
        self.mode = mode

        check_prototypes(config.n_proto, self.emb_dim, self.load_proto, config.proto_path)
        # This module contains the EM step
        self.panther = PANTHERBase(self.emb_dim, p=config.n_proto, L=config.em_iter,
                         tau=config.tau, out=config.out_type, ot_eps=config.ot_eps,
                         load_proto=config.load_proto, proto_path=config.proto_path,
                         fix_proto=config.fix_proto)

    def representation(self, x):
        """
        Construct unsupervised slide representation
        """
        out, qqs = self.panther(x)
        return {'repr': out, 'qq': qqs}

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
                data = data.to(next(self.parameters()).device)
            
            with torch.no_grad():
                out = self.representation(data)
                out = out['repr'].data.detach().cpu()

            X.append(out)

        X = torch.cat(X)

        return X, y