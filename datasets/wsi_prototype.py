import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import h5py


class WSIProtoDataset(Dataset):
    """WSI Custer Dataset."""
    def __init__(self,data_dir,csv_path):
        super().__init__()
        self.data_dir = data_dir
        df = pd.read_csv(csv_path)
        self.df = df
        self.df['case_id'] = self.df['case_id'].astype(str)
        self.idx2sample_df = pd.DataFrame({'sample_id': self.df['case_id'].astype(str).unique()})

        self.X = None
        self.y = None

    def __len__(self):
        return len(self.idx2sample_df)
    
    def get_sample_id(self, idx):
        return self.idx2sample_df.loc[idx]['sample_id']

    def get_feat_paths(self, idx):
        slide_ids = list(self.df.loc[self.df['case_id']==self.get_sample_id(idx), 'slide_id']) # type: ignore
        if isinstance(slide_ids, str):
            slide_ids = [slide_ids]
        feat_paths = [os.path.join(self.data_dir, str(slide_name) + '.h5') for slide_name in slide_ids]
        return feat_paths

    def __getitem__(self, idx):
        feat_paths = self.get_feat_paths(idx)

        # Read features (and coordinates, Optional) from pt/h5 file
        all_features = []
        all_coords = []
        for feat_path in feat_paths:
            with h5py.File(feat_path, 'r') as f:
                features = f['features'][:] # type: ignore
            all_features.append(features)
        all_features = torch.from_numpy(np.concatenate(all_features, axis=0))

        out = {'img': all_features,
               'coords': all_coords}

        return out