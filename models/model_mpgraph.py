import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, GCN2Conv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap, global_sort_pool as gsp, SAGPooling, GlobalAttention


class MPGraph_Surv(torch.nn.Module):
    def __init__(self, input_dim=1024, proto_dim=64, p=8, n_classes=1, mode='mg'):
        super().__init__()
        self.n_classes = n_classes
        self.p = p
        self.mode = mode
        hidden_dim = self.p * proto_dim

        # graph part
        self.graphconv = nn.Sequential(*[GraphConv(input_dim, hidden_dim), nn.ReLU()])
        self.graphlinear = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim//p), 
                          nn.ReLU(), 
                          nn.Dropout(0.25)) for i in range(p)])
        self.graph_rho = nn.Sequential(*[nn.Linear(hidden_dim,hidden_dim//2), nn.ReLU(), nn.Dropout(0.25)])

        # mlp part
        self.mlp = nn.Sequential(*[nn.Linear(input_dim,hidden_dim), nn.ReLU(), nn.Dropout(0.25)])
        self.mlplinear = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim//p), 
                          nn.ReLU(), 
                          nn.Dropout(0.25)) for i in range(p)])
        self.mlp_rho = nn.Sequential(*[nn.Linear(hidden_dim,hidden_dim//2), nn.ReLU(), nn.Dropout(0.25)])
        
        # classifier
        self.post = nn.Sequential(*[nn.Linear(hidden_dim,hidden_dim//2), nn.ReLU(), nn.Dropout(0.25)])
        self.classifier = torch.nn.Linear(hidden_dim//2, n_classes)

    def forward(self, data, return_feature=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # graph part
        h_graph = self.graphconv[0](x,edge_index,edge_attr)
        h_graph = self.graphconv[1:](h_graph)
        h_graph = h_graph.view(batch.max() + 1, self.p, -1)
        h_graph = torch.stack([self.graphlinear[idx](h_graph[:, idx, :]) for idx in range(self.p)], dim=1)
        h_graph = h_graph.reshape(batch.max() + 1, -1)
        h_graph = self.graph_rho(h_graph)

        # mlp part
        h_mlp = self.mlp(x)
        h_mlp = h_mlp.view(batch.max() + 1, self.p, -1)
        h_mlp = torch.stack([self.mlplinear[idx](h_mlp[:, idx, :]) for idx in range(self.p)], dim=1)
        h_mlp = h_mlp.reshape(batch.max() + 1, -1)
        h_mlp = self.mlp_rho(h_mlp)

        # classifier
        if self.mode == 'morpho':
            h = h_mlp
        elif self.mode == 'graph':
            h = h_graph
        elif self.mode == 'mg':
            h = torch.cat([h_graph, h_mlp], dim=1)
            h = self.post(h)
        
        # slide feature
        if return_feature:
            return h.squeeze()

        logits  = self.classifier(h)

        if self.n_classes == 4:
            Y_hat = torch.topk(logits, 1, dim = 1)[1] # logits needs to be a [1 x 4] vector
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        else:
            risk = torch.exp(logits).squeeze(1)
            return logits, risk