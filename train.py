#!/usr/bin/env python
# coding=utf-8
import torch
from torch.utils.tensorboard import SummaryWriter # type: ignore
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.loader import DataLoader as GraphDataLoader

import argparse
import os
import sys
sys.path.append('.')
import warnings
import datetime
import time
warnings.filterwarnings("ignore")

from datasets.survival import SurvDataset
from models.panther.model_panther import PANTHER
from models.panther.base_layers import PrototypeTokenizer
from models.panther.model_linear import IndivMLPEmb
from models.model_attmil import AttMIL
from models.model_trans import TransMIL
from models.model_ilra import ILRA
from models.model_gcn import DeepGraphConv_Surv, PatchGCN_Surv
from models.model_wikg import WiKG
from models.model_h2t import H2T
from models.model_mpgraph import MPGraph_Surv

from utils.file_utils import save_pkl, load_pkl
from utils.losses import NLLSurvLoss,CoxLoss,SurvRankingLoss
from utils.core_utils import main_process 


parser = argparse.ArgumentParser(description="MPGraph & Baseline Training Script")

# Data arguments
parser.add_argument("--data_dir", type=str, default="./unifeats", help="features directory path")
parser.add_argument("--train_csv_path", type=str, default="./train.csv", help="train csv path")
parser.add_argument("--val_csv_path", type=str, default="./val.csv", help="val csv path")
parser.add_argument("--test_csv_path", type=str, default="./test.csv", help="test csv path")
parser.add_argument('--fold', type=int, default=1, help='fold index')
parser.add_argument("--cache_dir", type=str, default="./data_cache", help="directory to store cached representations")

# Prototype arguments
parser.add_argument('--n_proto', type=int, default=8, help='Number of prototypes')
parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--model_type', type=str, default='PANTHER', help='type of embedding model')
parser.add_argument('--em_iter', type=int, default=1, help='Number of iterations for embedding')
parser.add_argument('--tau', type=float, default=1.0)
parser.add_argument('--out_type', type=str, default='allcat')
parser.add_argument('--ot_eps', default=1.0, type=float, help='Strength for entropic constraint regularization for OT')
parser.add_argument('--proto_path', type=str, default=None, help='path to load pre-clustered prototypes')

# Training arguments
parser.add_argument('--mode', type=str, choices=['mil', 'gcn', 'h2t', 'panther', 'mpgraph'], default='mpgraph')
parser.add_argument("--train_batch_size", type=int, default=8, help="train batch size")
parser.add_argument("--batch_size", type=int, default=1, help="val/test batch size")
parser.add_argument("--num_workers", type=int, default=12, help="num workers")
parser.add_argument("--num_epochs", type=int, default=50, help="num of epochs")
parser.add_argument("--save_model_path", type=str, default="./results/saved_models/model.pth", help="model saving path")
parser.add_argument("--use_tensorboard", action="store_true", help="whether to use tensorboard")
parser.add_argument("--logdir", type=str, default="./logs", help="tensorboard log dir")

# Optimizer arguments
parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW"], help="optimizer type")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")

args = parser.parse_args()


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(args)

    print("========Preparing Dataset========")
    if args.mode == 'mil':
        train_dataset = SurvDataset(args.data_dir, args.train_csv_path, mode=args.mode, num_proto=args.n_proto)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

        #================= sampler for NLL_loss=================#
        # sample_label,counts = train_dataset.value_counts()
        # weights = 1./ torch.tensor(counts, dtype=torch.float)
        # samples_weights = weights[sample_label]
        # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True) # type: ignore
        # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers)
        
        val_dataset = SurvDataset(args.data_dir,args.val_csv_path, mode=args.mode, num_proto=args.n_proto)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        test_dataset = SurvDataset(args.data_dir,args.test_csv_path, mode=args.mode, num_proto=args.n_proto)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    elif args.mode == 'gcn':
        train_dataset = SurvDataset(args.data_dir, args.train_csv_path, mode=args.mode, num_proto=args.n_proto)
        train_loader = GraphDataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers) # type: ignore
        
        #================= sampler for NLL_loss=================#
        # sample_label,counts = train_dataset.value_counts()
        # weights = 1./ torch.tensor(counts, dtype=torch.float)
        # samples_weights = weights[sample_label]
        # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True) # type: ignore
        # train_loader = GraphDataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, sampler=sampler, num_workers=args.num_workers) # type: ignore
        
        val_dataset = SurvDataset(args.data_dir,args.val_csv_path, mode=args.mode, num_proto=args.n_proto)
        val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) # type: ignore

        test_dataset = SurvDataset(args.data_dir,args.test_csv_path, mode=args.mode, num_proto=args.n_proto)
        test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) # type: ignore
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    elif args.mode == 'h2t':
        train_dataset = SurvDataset(args.data_dir, args.train_csv_path, mode=args.mode, num_proto=args.n_proto)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

        val_dataset = SurvDataset(args.data_dir, args.val_csv_path, mode=args.mode, num_proto=args.n_proto)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        test_dataset = SurvDataset(args.data_dir, args.test_csv_path, mode=args.mode, num_proto=args.n_proto)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
        # Construct unsupervised slide-level embedding
        embeddings_fpath = os.path.join(args.cache_dir, f"h2t_fold{args.fold}_slide_representation.pkl")
        if os.path.isfile(embeddings_fpath):
            embeddings = load_pkl(embeddings_fpath)
            for k, loader in dataloaders.items():
                print(f'\n\tEmbedding already exists! Loading {k}', end=' ')
                loader.dataset.X, loader.dataset.y = embeddings[k]['X'], embeddings[k]['y'] # type: ignore
        else:
            embed = H2T(config=args, mode='emb').to(device)
            embeddings = {}
            for key, loader in dataloaders.items():
                print(f"\nAggregating {key} set features...")
                X, y = embed.predict_emb(loader.dataset, use_cuda=torch.cuda.is_available())
                loader.dataset.X, loader.dataset.y = X, y # type: ignore
                embeddings[key] = {'X': X, 'y': y}
            save_pkl(embeddings_fpath, embeddings)
        print("\nSlide embedding construction finished!")  

    elif args.mode == 'panther':
        train_dataset = SurvDataset(args.data_dir, args.train_csv_path, mode=args.mode, num_proto=args.n_proto)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

        val_dataset = SurvDataset(args.data_dir, args.val_csv_path, mode=args.mode, num_proto=args.n_proto)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        test_dataset = SurvDataset(args.data_dir, args.test_csv_path, mode=args.mode, num_proto=args.n_proto)   
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
        # Construct unsupervised slide-level embedding
        embeddings_fpath = os.path.join(args.cache_dir, f"fold{args.fold}_slide_representation.pkl")
        if os.path.isfile(embeddings_fpath):
            embeddings = load_pkl(embeddings_fpath)
            for k, loader in dataloaders.items():
                print(f'\n\tEmbedding already exists! Loading {k}', end=' ')
                loader.dataset.X, loader.dataset.y = embeddings[k]['X'], embeddings[k]['y'] # type: ignore
        else:
            embed = PANTHER(config=args, mode='emb').to(device)
            embeddings = {}
            for key, loader in dataloaders.items():
                print(f"\nAggregating {key} set features...")
                X, y = embed.predict_emb(loader.dataset, use_cuda=torch.cuda.is_available())
                loader.dataset.X, loader.dataset.y = X, y # type: ignore
                embeddings[key] = {'X': X, 'y': y}
            save_pkl(embeddings_fpath, embeddings)
        
        # Construct tokenized slide-level embedding
        if args.out_type == 'allcat':
            print("\nGenerting Tokenized slide embeddings..")
            tokenizer = PrototypeTokenizer(args.model_type, args.out_type, args.n_proto)
            embeddings = {}
            for k, loader in dataloaders.items():
                prob, mean, cov = tokenizer(loader.dataset.X) # type: ignore
                loader.dataset.X = {'prob': prob, 'mean': mean, 'cov': cov} # type: ignore
                embeddings[k] = {'prob': prob, 'mean': mean, 'cov': cov}
            fpath_new = os.path.join(args.cache_dir, f"fold{args.fold}_tokenized_slide_representation.pkl")
            save_pkl(fpath_new, embeddings)

        print("\nSlide embedding construction finished!")

    elif args.mode == 'mpgraph':
        train_dataset = SurvDataset(args.data_dir, args.train_csv_path, mode=args.mode, num_proto=args.n_proto)
        train_loader = GraphDataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers) # type: ignore

        val_dataset = SurvDataset(args.data_dir, args.val_csv_path, mode=args.mode, num_proto=args.n_proto)
        val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) # type: ignore

        test_dataset = SurvDataset(args.data_dir, args.test_csv_path, mode=args.mode, num_proto=args.n_proto)   
        test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) # type: ignore
        dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
        
        # Construct unsupervised slide-level embedding
        embeddings_fpath = os.path.join(args.cache_dir, f"{args.n_proto}_fold{args.fold}_slide_representation.pkl")
        if os.path.isfile(embeddings_fpath):
            embeddings = load_pkl(embeddings_fpath)
            for k, loader in dataloaders.items():
                print(f'\n\tEmbedding already exists! Loading {k}', end=' ')
                loader.dataset.X, loader.dataset.y = embeddings[k]['X'], embeddings[k]['y'] # type: ignore
        else:
            embed = PANTHER(config=args, mode='emb').to(device)
            embeddings = {}
            for key, loader in dataloaders.items():
                print(f"\nAggregating {key} set features...")
                X, y = embed.predict_emb(loader.dataset, use_cuda=torch.cuda.is_available())
                loader.dataset.X, loader.dataset.y = X, y # type: ignore
                embeddings[key] = {'X': X, 'y': y}
            save_pkl(embeddings_fpath, embeddings)
        
        # Construct tokenized slide-level embedding
        if args.out_type == 'allcat':
            print("\nGenerting Tokenized slide embeddings..")
            tokenizer = PrototypeTokenizer(args.model_type, args.out_type, args.n_proto)
            embeddings = {}
            for k, loader in dataloaders.items():
                prob, mean, cov = tokenizer(loader.dataset.X) # type: ignore
                loader.dataset.X = {'prob': prob, 'mean': mean, 'cov': cov} # type: ignore
                embeddings[k] = {'prob': prob, 'mean': mean, 'cov': cov}
            fpath_new = os.path.join(args.cache_dir, f"{args.n_proto}_fold{args.fold}_tokenized_slide_representation.pkl")
            save_pkl(fpath_new, embeddings)

        print("\nSlide embedding construction finished!")    
    print("========Dataset Done========")

    print("========Preparing Model========")
    if args.mode == 'mil':
        model = AttMIL(n_classes=4)
        # model = TransMIL(n_classes=4)
        # model = WiKG(n_classes=4)
        # model = ILRA(n_classes=4)
    elif args.mode == 'gcn':
        model = DeepGraphConv_Surv(n_classes=4)
        # model = PatchGCN_Surv(n_classes=4)
    elif args.mode == 'h2t':
        model = IndivMLPEmb(n_classes=1, p=args.n_proto, in_dim=1024)
    elif args.mode == 'panther':
        model = IndivMLPEmb(n_classes=1, p=args.n_proto, in_dim=2049)
    elif args.mode == 'mpgraph':
        model = MPGraph_Surv(n_classes=1, p=args.n_proto)
    model.to(device)
    print(model)
    print("========Model Done========")
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    params = [p for p in model.parameters() if p.requires_grad]

    # criterion = NLLSurvLoss(alpha=0.25)
    # criterion = CoxLoss()
    criterion = SurvRankingLoss()
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    print("========Start Training========")
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    start_time = time.time()   
    main_process(model=model, 
                 criterion=criterion, 
                 optimizer=optimizer, 
                 lr_scheduler=lr_scheduler,
                 dataloaders=dataloaders, 
                 writer=writer,
                 num_epochs=args.num_epochs, 
                 use_tensorboard=args.use_tensorboard,
                 device=device,
                 save_model_path=args.save_model_path)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))    


if __name__ == "__main__":
    main(args)