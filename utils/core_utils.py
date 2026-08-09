#!/usr/bin/env python
# coding=utf-8

import os
import copy
import torch
import numpy as np
import pandas as pd

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
from tqdm import tqdm

from losses import NLLSurvLoss


def l1_reg_all(model):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """learning rate warmup"""

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class Monitor_CIndex(EarlyStopping):
    """Early stops the training if cindex doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        super().__init__()
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)


def train_loop_survival(epoch, model, loader, optimizer, lr_scheduler, device, writer=None, loss_fn=None, reg_fn=None, lambda_reg=1e-4, gc=16):
    '''one epoch training'''   
    model.train()
    train_loss = 0.

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, batch in enumerate(tqdm(loader)):
        label = batch['label'].to(device)
        c = batch['censor'].to(device)
        hazards, S, Y_hat, _, _ = model(batch['img'].to(device)) # return hazards, S, Y_hat, A_raw, results_dict       
        risk = -torch.sum(S, dim=1)
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c) # type: ignore

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss.item(), label.item(), float(batch['time'].item()), float(risk), batch['img'].size(0)))
        
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        
        all_risk_scores.extend(risk.detach().cpu().numpy().tolist()) 
        all_censorships.extend(c.cpu().numpy().tolist())
        all_event_times.extend(batch['time'].cpu().numpy().tolist())

        train_loss += loss.item() + loss_reg

        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_survival(epoch, model, loader, device, writer=None, loss_fn=None, reg_fn=None, lambda_reg=1e-4):
    '''validation after each epoch'''
    model.eval()
    val_loss = 0.
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, batch in enumerate(tqdm(loader)):
        label = batch['label'].to(device)
        c = batch['censor'].to(device)

        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(batch['img'].to(device)) # return hazards, S, Y_hat, A_raw, results_dict       
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c) # type: ignore
            risk = -torch.sum(S, dim=1)

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        
        all_risk_scores.extend(risk.cpu().numpy().tolist())
        all_censorships.extend(c.cpu().numpy().tolist())
        all_event_times.extend(batch['time'].cpu().numpy().tolist())

        val_loss += loss.item() + loss_reg

    val_loss /= len(loader)
    c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08)[0]
    
    print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, c_index))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    return val_loss, c_index


def summary_survival(model, loader, device):
    '''test after training'''
    model.eval()

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    all_patients = []
    all_labels = []

    patient_results = {}

    for batch_idx, batch in enumerate(tqdm(loader)):

        with torch.no_grad():
            hazards, S, Y_hat, _, _ = model(batch['img'].to(device)) # return hazards, S, Y_hat, A_raw, results_dict       
            risk = -torch.sum(S, dim=1)
       
        all_risk_scores.extend(risk.cpu().numpy().tolist())
        all_censorships.extend(batch['censor'].numpy().tolist())
        all_event_times.extend(batch['time'].numpy().tolist())
        all_patients.extend(batch['slide_name'])
        all_labels.extend(batch['label'].numpy().tolist())
    
    # ===== C-index ===== #
    c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08)[0]
    print('Test C-index: {:.4f}'.format(c_index))
    
    # ===== iAUC ===== #
    y_surv = Surv.from_arrays(event=(1 - np.array(all_censorships)).astype(bool),time=np.array(all_event_times))
    event_times = np.array(all_event_times)[(1 - np.array(all_censorships)) == 1]
    t_lower = np.quantile(event_times, 0.20)
    t_upper = np.quantile(event_times, 0.81)
    eval_times = np.linspace(t_lower, t_upper, 10)
    auc_times, auc_values = cumulative_dynamic_auc(y_surv,y_surv,np.array(all_risk_scores),eval_times)
    restricted_iAUC = np.mean(auc_values)
    print(f"Test iAUC: {restricted_iAUC:.4f}")

    patient_results = pd.DataFrame({'patient': all_patients, 
                                    'risk': all_risk_scores, 
                                    'disc_label': all_labels, 
                                    'survival': all_event_times, 
                                    'censorship': all_censorships})

    return patient_results, c_index


def train_cox_survival(epoch, model, loader, optimizer, lr_scheduler, device, writer=None, loss_fn=None):
    model.train()
    train_loss = 0.

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, batch in enumerate(tqdm(loader)):
        c = batch['censor'].to(device)
        event_time = batch['time'].to(device)
        logits, risk = model(batch['img'].to(device))       
        loss = loss_fn(logits, event_time, c.float()) # type: ignore       

        all_risk_scores.extend(risk.detach().cpu().numpy().tolist()) 
        all_censorships.extend(c.cpu().numpy().tolist())
        all_event_times.extend(event_time.cpu().numpy().tolist())

        train_loss += loss.item()
        if (batch_idx+1) % 5 == 0:
            print('batch {}, loss: {:.4f}'.format(batch_idx, loss.item()))
            unique_values, counts = torch.unique(c, return_counts=True)
            print(f'Value: {unique_values.cpu().numpy()}, Count: {counts.cpu().numpy()}')

        # backward pass
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    # c_index = concordance_index(all_event_times, all_risk_scores, event_observed=1-all_censorships) 
    c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores))[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index, epoch)


def validate_cox_survival(epoch, model, loader, device, writer=None, loss_fn=None):
    model.eval()
    val_loss = 0.
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, batch in enumerate(tqdm(loader)):
        c = batch['censor'].to(device)
        event_time = batch['time'].to(device)

        with torch.no_grad():
            logits, risk = model(batch['img'].to(device))       
            loss = loss_fn(logits, event_time, c.float()) # type: ignore
        
        all_risk_scores.extend(risk.cpu().numpy().tolist())
        all_censorships.extend(c.cpu().numpy().tolist())
        all_event_times.extend(event_time.cpu().numpy().tolist())

        val_loss += loss.item()

    val_loss /= len(loader)
    c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08)[0]
    
    print('Epoch: {}, val_loss: {:.4f}, val_c_index: {:.4f}'.format(epoch, val_loss, c_index))
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    return val_loss, c_index


def summary_cox_survival(model, loader, device):
    model.eval()

    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    all_patients = []
    all_labels = []

    patient_results = {}

    for batch_idx, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            logits, risk = model(batch['img'].to(device))       

        all_risk_scores.extend(risk.cpu().numpy().tolist())
        all_censorships.extend(batch['censor'].numpy().tolist())
        all_event_times.extend(batch['time'].numpy().tolist())
        all_patients.extend(batch['slide_name'])
        all_labels.extend(batch['label'].numpy().tolist())
    
    # ===== C-index ===== #
    c_index = concordance_index_censored((1-np.array(all_censorships)).astype(bool), np.array(all_event_times), np.array(all_risk_scores), tied_tol=1e-08)[0]
    print('Test C-index: {:.4f}'.format(c_index))

    # ===== iAUC ===== #
    y_surv = Surv.from_arrays(event=(1 - np.array(all_censorships)).astype(bool),time=np.array(all_event_times))
    event_times = np.array(all_event_times)[(1 - np.array(all_censorships)) == 1]
    t_lower = np.quantile(event_times, 0.20)
    t_upper = np.quantile(event_times, 0.81)
    eval_times = np.linspace(t_lower, t_upper, 10)
    # cumulative/dynamic AUC
    auc_times, auc_values = cumulative_dynamic_auc(y_surv,y_surv,np.array(all_risk_scores),eval_times)
    restricted_iAUC = np.mean(auc_values)
    print(f"Test iAUC: {restricted_iAUC:.4f}")

    patient_results = pd.DataFrame({'patient': all_patients, 
                                    'risk': all_risk_scores, 
                                    'disc_label': all_labels, 
                                    'survival': all_event_times, 
                                    'censorship': all_censorships})
    
    return patient_results, c_index



def main_process(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloaders: Dict[str, DataLoader], # type: ignore
    num_epochs: int,
    use_tensorboard: bool,
    device: torch.device,
    save_model_path: str,
    writer=None
):
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    
    model.train()
    trigger_times = 0
    best_score = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    reg_fn = None
    # reg_fn = l1_reg_all

    is_nll_loss = isinstance(criterion, NLLSurvLoss)

    for epoch in range(num_epochs):
        if is_nll_loss:
            train_loop_survival(epoch, model, dataloaders["train"], optimizer, lr_scheduler, device, writer, criterion, reg_fn=reg_fn)
            _,c_index = validate_survival(epoch, model, dataloaders["val"], device, writer, criterion, reg_fn=reg_fn)

            if epoch < 5:
                pass
            elif c_index > best_score:
                best_score = c_index
                best_epoch = epoch
                trigger_times = 0
                best_state_dict = copy.deepcopy(model.state_dict())            
            else:
                trigger_times += 1
                print(f'Early stopping counter: {trigger_times}/{20}')
                if trigger_times >= 20:
                    print("Early stopping!")
                    break
        else:
            train_cox_survival(epoch, model, dataloaders["train"], optimizer, lr_scheduler, device, writer, criterion)
            _,c_index = validate_cox_survival(epoch, model, dataloaders["val"], device, writer, criterion)

            if epoch < 5:
                pass
            elif c_index > best_score:
                best_score = c_index
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())

    print("Training Done!")
    print(f"Best Valid C-index: {best_score:.4f} at epoch {best_epoch}")
    torch.save(best_state_dict, save_model_path)

    print("========Start Testing========")
    model.load_state_dict(best_state_dict)
    if is_nll_loss:
        df, test_c_index = summary_survival(model, dataloaders["test"], device)
    else:
        df, test_c_index = summary_cox_survival(model, dataloaders["test"], device)

    if use_tensorboard:
        writer.close() # type: ignore
    
    # Save test prediction results
    df.sort_values(by=['patient'], inplace=True, ascending=False)
    output_csv_path = os.path.join(os.path.dirname(save_model_path), 'test_result.csv')
    df.to_csv(output_csv_path, sep=',', index=False)
    print(f"Test results successfully exported to: {output_csv_path}")