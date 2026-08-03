import os 
import sys

sys.path.append('..')
import torch
import h5py
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch_geometric.data import Data

from datasets.batchWSI import BatchWSI
from utils.graph_utils import wasserstein_distance, symmetric_normalization


class SurvDataset(Dataset):
    """
    Dataset for WSIs-based survival prediction.

    Args:
        data_dir (str): Directory containing feature files.
        csv_path (str): Path to metadata CSV.
        mode (str): Dataset mode ('mil', 'gcn', 'panther', 'h2t', 'mpgraph').
        num_proto (int): Number of prototypes.
    """
    def __init__(self, data_dir, csv_path, mode, num_proto):
        super().__init__()
        self.mode = mode
        self.num_proto = num_proto
        self.data_dir = data_dir
        df = pd.read_csv(csv_path)
        self.df = df
        self.df['case_id'] = self.df['case_id'].astype(str)
        self.idx2sample_df = pd.DataFrame({'sample_id': self.df['case_id'].astype(str).unique()})

        self.event_times = []
        self.status = []
        self.dislabels = []
        for idx in self.idx2sample_df.index:
            event_time, status, dislabel = self.get_labels(idx)
            self.event_times.append(event_time)
            self.status.append(status)
            self.dislabels.append(dislabel)
        
        # construct and store prob, mean, cov every train
        self.X = None
        self.y = None

    def __len__(self):
        return len(self.idx2sample_df)
    
    def get_sample_id(self, idx) -> str:
        return self.idx2sample_df.loc[idx]['sample_id']
    
    def get_labels(self, idx):
        labels = self.df.loc[self.df['case_id']==self.get_sample_id(idx), ['time', 'status', 'dislabel']]
        if isinstance(labels, pd.Series):
            labels = list(labels)
        elif isinstance(labels, pd.DataFrame):
            labels = list(labels.iloc[0])
        return labels
    
    def get_adj(self, idx) -> torch.Tensor:
        """
        Construct prototype graph adjacency matrix.

        Edge weights are calculated according to
        Wasserstein distances between Gaussian prototypes.
        """
        prob, mean, cov = self.X['prob'][idx], self.X['mean'][idx], self.X['cov'][idx] # type: ignore
        adj = torch.zeros((self.num_proto,self.num_proto))
        for i in range(self.num_proto):
            for j in range(i + 1, self.num_proto):  
                dist = wasserstein_distance(mean[i], cov[i], mean[j], cov[j])
                adj[i, j] = 1/(dist + 1e-10)
                adj[j, i] = 1/(dist + 1e-10)
        adj = symmetric_normalization(adj)
        for i in range(self.num_proto):
            adj[i, i] = prob[i]
        return adj
    
    def get_feat_paths(self, idx):
        slide_ids = list(self.df.loc[self.df['case_id']==self.get_sample_id(idx), 'slide_id']) # type: ignore
        if isinstance(slide_ids, str):
            slide_ids = [slide_ids]
        feat_paths = [os.path.join(self.data_dir, str(slide_name) + '.h5') for slide_name in slide_ids]
        return feat_paths
    
    def get_gcn_paths(self, idx):
        slide_ids = list(self.df.loc[self.df['case_id']==self.get_sample_id(idx), 'slide_id']) # type: ignore
        if isinstance(slide_ids, str):
            slide_ids = [slide_ids]
        feat_paths = [os.path.join(self.data_dir.replace('unifeats','gcndata'), str(slide_name) + '.pt') for slide_name in slide_ids]
        return feat_paths
    
    def get_emb(self,idx):
        data = torch.cat([torch.Tensor(self.X['prob']).unsqueeze(dim=-1), torch.Tensor(self.X['mean']), torch.Tensor(self.X['cov'])], dim=-1) # type: ignore
        return data[idx]
    
    def __getitem__(self, index):
        label = int(self.dislabels[index])
        event_time = self.event_times[index]
        c = 1 - self.status[index]  # 1 for censored, 0 for uncensored
        slide_name = str(self.get_sample_id(index))        
        
        if self.X is not None:
            if self.mode == 'mpgraph':
                adj = self.get_adj(index)
                edge_index = torch.nonzero(adj, as_tuple=False).t()
                edge_weight = adj[edge_index[0], edge_index[1]].type(torch.float32)
                features = Data(x=self.X['mean'][index], edge_index=edge_index, edge_attr=edge_weight)
                out = {'img': features, 'coords': [], 'label': label, 'time': event_time, 'censor': c, 'slide_name': slide_name}
                return out
            
            elif self.mode == 'panther':
                features = self.get_emb(index)
                out = {'img': features, 'coords': [], 'label': label, 'time': event_time, 'censor': c, 'slide_name': slide_name}
                return out
            
            elif self.mode == 'h2t':
                features = self.X[index]
                out = {'img': features, 'coords': [], 'label': label, 'time': event_time, 'censor': c, 'slide_name': slide_name}
                return out
        
        else: 
            if self.mode == 'gcn':
                feat_paths = self.get_gcn_paths(index)
                all_features = []
                for feat_path in feat_paths:
                    features = torch.load(feat_path)
                    all_features.append(features)
                all_features = BatchWSI.from_data_list(all_features, update_cat_dims={'edge_latent': 1})
                out = {'img': all_features, 'coords': [], 'label': label, 'time': event_time, 'censor': c, 'slide_name': slide_name}
                return out
            
            else :
                feat_paths = self.get_feat_paths(index)
                all_features = []
                all_coords = []
                for feat_path in feat_paths:
                    with h5py.File(feat_path, 'r') as f:
                        features = f['features'][:] # type: ignore
                    all_features.append(features)
                all_features = torch.from_numpy(np.concatenate(all_features, axis=0))
                out = {'img': all_features, 'coords': all_coords, 'label': label, 'time': event_time, 'censor': c, 'slide_name': slide_name}
                return out
    
    def value_counts(self):
        label_list = self.dislabels
        counts = self.df.drop_duplicates('case_id').value_counts('dislabel')
        desired_order = [0, 1, 2, 3]
        counts = list(counts.reindex(desired_order, fill_value=0))
        return label_list,counts
