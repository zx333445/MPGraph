import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):
    ''''''
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # type: ignore # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)  # type: ignore # [N,256]
        b = self.attention_b(x)  # type: ignore # [N,256]
        A = a.mul(b)             # 逐元素相乘 [N,256]
        A = self.attention_c(A)  # [N, ncls]
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""

class AttMIL(nn.Module):
    def __init__(self, gate = True, size_arg = "small", dropout = True, n_classes=4):
        super().__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(0.25))
        self.classifiers = nn.Linear(size[2], n_classes)
        self.n_classes = n_classes

        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
    
    def forward(self, x, attention_only=False, return_feature=False):
        device = x.device                # B,N,1024  
        A, x = self.attention_net(x)     # B,N,1        
        A = A.permute(0, 2, 1)           # B,1,N
        if attention_only:
            return A
        
        A = F.softmax(A, dim=2)  # softmax over N
        M = torch.bmm(A, x) 
        hpath = self.rho(M)
        if return_feature:
            return hpath.squeeze()
        
        logits  = self.classifiers(hpath).squeeze(1) # logits needs to be a [1 x 4] vector
        
        if self.n_classes == 4:
            Y_hat = torch.topk(logits, 1, dim = 1)[1] # logits needs to be a [1 x 4] vector
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        else:
            risk = torch.exp(logits).squeeze(1)
            return logits, risk
