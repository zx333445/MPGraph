import torch
import torch.nn as nn


class LinearEmb(nn.Module):
    """
    Linear fully-connected layer from slide representation to output
    """
    def __init__(self, 
                 in_dim=8,
                 n_classes=1):
        super().__init__()
        self.classifier = nn.Linear(in_dim, n_classes, bias=False)
        self.n_classes = n_classes

    def forward(self, h):
        logits = self.classifier(h)
        if self.n_classes == 4:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        else:
            risk = torch.exp(logits).squeeze(1)
            return logits, risk


def create_mlp_with_dropout(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias)) # type: ignore
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim)) # type: ignore
        if not end_with_fc:
            layers.append(act)
            layers.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*layers)
    return mlp

#
# MLP per prototype
#
class IndivMLPEmb(nn.Module):
    """
    Comprised of three MLP (in sequence), each of which can be enabled/disabled and configured accordingly
    - Shared: Shared MLP across prototypes for feature dimension reduction
    - Indiv: Individual MLP per prototype
    - Post: Shared MLP across prototypes for final feature dimension reduction
    """
    def __init__(self, 
                    in_dim: int = 2049,
                    n_classes: int = 4,
                    shared_embed_dim: int = 512,
                    indiv_embed_dim: int = 64,
                    postcat_embed_dim: int = 256,
                    shared_mlp: bool = True,
                    indiv_mlps: bool = True,
                    postcat_mlp: bool = True,
                    n_fc_layers: int = 1,
                    shared_dropout: float = 0.25,
                    indiv_dropout: float = 0.25,
                    postcat_dropout: float = 0.25,
                    p: int = 8):
        super().__init__()
        self.n_classes = n_classes
        self.p = p
        mlp_func = create_mlp_with_dropout

        if shared_mlp:
            self.shared_mlp = mlp_func(in_dim=in_dim,
                                    hid_dims=[shared_embed_dim] *(n_fc_layers - 1),
                                    dropout=shared_dropout,
                                    out_dim=shared_embed_dim,
                                    end_with_fc=False)
            next_in_dim = shared_embed_dim
        else:
            self.shared_mlp = nn.Identity()
            next_in_dim = in_dim

        if indiv_mlps:
            self.indiv_mlps = nn.ModuleList([mlp_func(in_dim=next_in_dim,
                                hid_dims=[indiv_embed_dim] *
                                        (n_fc_layers - 1),
                                dropout=indiv_dropout,
                                out_dim=indiv_embed_dim,
                                end_with_fc=False) for i in range(p)])
            next_in_dim = p * indiv_embed_dim
        else:
            self.indiv_mlps = nn.ModuleList([nn.Identity() for i in range (p)])
            next_in_dim = p * next_in_dim

        if postcat_mlp:
            self.postcat_mlp = mlp_func(in_dim=next_in_dim,
                                    hid_dims=[postcat_embed_dim] *
                                            (n_fc_layers - 1),
                                    dropout=postcat_dropout,
                                    out_dim=postcat_embed_dim,
                                    end_with_fc=False)
            next_in_dim = postcat_embed_dim
        else:
            self.postcat_mlp = nn.Identity()

        self.classifier = nn.Linear(next_in_dim, n_classes, bias=False)

    def forward(self, h, attn_mask=None, return_feature=False):
        h = self.shared_mlp(h)
        h = torch.stack([self.indiv_mlps[idx](h[:, idx, :]) for idx in range(self.p)], dim=1)
        h = h.reshape(h.shape[0], -1)   # (n_samples, n_proto * config.indiv_embed_dim)
        h = self.postcat_mlp(h)
        if return_feature:
            return h.squeeze()
        logits = self.classifier(h)

        if self.n_classes == 4:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            # risk = -torch.sum(S, dim=1)
            return hazards, S, Y_hat, None, None
        else:
            risk = torch.exp(logits).squeeze(1)
            return logits, risk


if __name__ == '__main__':

    model = IndivMLPEmb(n_classes=1)
    a = torch.randn(4, 8, 2049)
    time = torch.tensor([19, 21, 33, 40.])
    cencor = torch.tensor([1, 1, 0, 1])
    import sys
    sys.path.append('..')
    from utils.losses import SurvRankingLoss
    lossfn = SurvRankingLoss()
    import ipdb;ipdb.set_trace()