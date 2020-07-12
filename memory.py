import os
import math
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from models import l2norm

## Memory 
class Memory(nn.Module):    
    def __init__(self, mem_size=500000, feat_dim=256, margin=1, topk=1000, update_rate=0.1):
        super(Memory, self).__init__()
        self.mem_size = mem_size
        self.feat_dim = feat_dim            
        self.Mem = nn.Parameter(torch.zeros(mem_size, feat_dim))
        self.Ages = nn.Parameter(torch.zeros(mem_size, 1))
        self.topk = topk
        self.margin = margin
        self.update_rate = update_rate
        # At this time, we don't train mem by gradient descent
        self.Mem.requires_grad = False
        self.Ages.requires_grad = False
    
    def update_mem(self, x, labels):
        with torch.no_grad():
            self.Mem[labels] = l2norm(self.update_rate * x.data + (1 - self.update_rate) * self.Mem[labels])

    def update_mem_with_ages(self, x, labels):        
        with torch.no_grad():
            self.Ages[labels] += 1.
            self.Mem[labels] = l2norm(x.data + self.Mem[labels] * self.Ages[labels]) 
            
    def search_l2(self, x, topk):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.mem_size) + \
                  torch.pow(self.Mem, 2).sum(dim=1, keepdim=True).expand(self.mem_size, batch_size).t()
        distmat.addmm_(x, self.Mem.t(), beta=1, alpha=-2)
        distances, indices = torch.topk(distmat, topk, largest=False)
        return distances, indices

    def compute_l2loss(self, x, labels):
        """ L2 Distance
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """        
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.mem_size) + \
                  torch.pow(self.Mem, 2).sum(dim=1, keepdim=True).expand(self.mem_size, batch_size).t()
        distmat.addmm_(x, self.Mem.t(), beta=1, alpha=-2)
        classes = torch.arange(self.mem_size).long()
        if labels.is_cuda: 
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.mem_size)
        mask = labels.eq(classes.expand(batch_size, self.mem_size))

        dist1 = distmat * mask.float()
        min_loss = dist1.clamp(min=1e-12, max=1e+12).sum(1)

        dist2 = distmat * (1.0 - mask.float())
        max_loss = torch.topk(dist2, self.topk, dim=1, largest=False)[0].sum(1) /  (self.topk - 1)
        loss = F.relu(min_loss - max_loss + self.margin)
        return loss.mean(), min_loss.mean(), max_loss.mean()
