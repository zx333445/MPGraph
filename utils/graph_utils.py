import torch


def wasserstein_distance(mean1, cov1_diag, mean2, cov2_diag):
    mean_diff = torch.norm(mean1 - mean2)
    trace_term = torch.sum(cov1_diag + cov2_diag - 2 * torch.sqrt(cov1_diag * cov2_diag))
    wasserstein_dist = mean_diff**2 + trace_term
    return wasserstein_dist


def kl_divergence(mean1, cov1_diag, mean2, cov2_diag):
    mean_diff = mean2 - mean1
    det_ratio = torch.sum(torch.log(cov2_diag / cov1_diag))
    trace_term = torch.sum(cov1_diag / cov2_diag)
    quadratic_term = torch.sum(mean_diff**2 / cov2_diag)
    
    kl_div = 0.5 * (trace_term + quadratic_term - len(mean1) + det_ratio)
    return kl_div


def symmetric_normalization(adj):    
    # 计算度矩阵
    D = torch.diag(torch.pow(adj.sum(dim=1), -0.5))
    
    # 计算对称归一化的邻接矩阵
    norm_adj = D @ adj @ D
    
    return norm_adj