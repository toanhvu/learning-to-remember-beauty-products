import os
import math
import numpy as np
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models import resnet50, mobilenet_v2, densenet121, resnet34

## Normalization classes and functions
class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x

def l2norm(x):
    eps = 1e-10
    norm = torch.sqrt(torch.sum(x * x, dim = 1) + eps)
    x= x / norm.unsqueeze(-1).expand_as(x)
    return x

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x

def l1norm(x):
    eps = 1e-10
    norm = torch.sum(torch.abs(x), dim = 1) + eps
    x= x / norm.expand_as(x)
    return x

## 
class Dense121(nn.Module):
    def __init__(self, feat_dim=256, pretrained=True, frozen=False):
        super(Dense121, self).__init__()         
        self.backbone = densenet121(pretrained=pretrained).features
        self.feature = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(1024, feat_dim, kernel_size=1),
                            nn.BatchNorm2d(feat_dim))
        
        self.attention = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(1024, 1, kernel_size=1))
        
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward_train(self, x, mask, soft_train=True):
        x = self.backbone(x)        
        feat = self.feature(x)     # B x feat_dim x H x W        

        att = torch.sigmoid(self.attention(x))
        
        mask = torch.nn.functional.interpolate(mask, size=att.shape[-2:], mode='nearest')        
        mask = (mask > 0).float()        

        if soft_train:
            feat = feat * mask.expand_as(feat)  
        else:
            feat = feat * att.expand_as(feat)  

        feat = feat.sum(dim=(2,3))
        feat = l2norm(feat)
        return feat, att, mask

    def forward(self, x, threshold=None):
        x = self.backbone(x)        
        feat = self.feature(x)     # B x feat_dim x H x W        

        att = torch.sigmoid(self.attention(x))
        if threshold:
            att = (att >= threshold).float()
        feat = feat * att.expand_as(feat)        
        feat = feat.sum(dim=(2,3))
        feat = l2norm(feat)
        return feat, att
    
class Dense121Edge(nn.Module):
    def __init__(self, feat_dim=256, pretrained=True, frozen=False):
        super(Dense121Edge, self).__init__()         
        pretrain_net = densenet121(pretrained=pretrained).features     
        # init first conv
        conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            conv1.weight[:,:-1,:,:] = pretrain_net[0].weight.data 
            nn.init.xavier_uniform_(conv1.weight[:,-1,:,:])
        pretrain_net[0] = conv1
        self.backbone = pretrain_net
        self.feature = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(1024, feat_dim, kernel_size=1),
                            nn.BatchNorm2d(feat_dim))
        
        self.attention = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1),
                            nn.BatchNorm2d(1024),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(1024, 1, kernel_size=1))
        
        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward_train(self, x, mask, soft_train=True):
        x = self.backbone(x)        
        feat = self.feature(x)     # B x feat_dim x H x W        

        att = torch.sigmoid(self.attention(x))
        
        mask = torch.nn.functional.interpolate(mask, size=att.shape[-2:], mode='nearest')        
        mask = (mask > 0).float()        

        if soft_train:
            feat = feat * mask.expand_as(feat)  
        else:
            feat = feat * att.expand_as(feat)  

        feat = feat.sum(dim=(2,3))
        feat = l2norm(feat)
        return feat, att, mask

    def forward(self, x, threshold=None):
        x = self.backbone(x)        
        feat = self.feature(x)     # B x feat_dim x H x W        

        att = torch.sigmoid(self.attention(x))
        if threshold:
            att = (att >= threshold).float()
        feat = feat * att.expand_as(feat)        
        feat = feat.sum(dim=(2,3))
        feat = l2norm(feat)
        return feat, att
    
        
        

class MultiScaleDense121(nn.Module):
    # ref: https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/modules/segmentation_body.py    
    def __init__(self, feat_dim=256, pretrained=True, frozen=False):
        super(MultiScaleDense121, self).__init__()         
        pretrain_net = densenet121(pretrained=pretrained).features                
        self.backbone = nn.ModuleList()
        self.backbone.append(pretrain_net[0:5])
        self.backbone.append(pretrain_net[5:7])
        self.backbone.append(pretrain_net[7:9])
        self.backbone.append(pretrain_net[9:])

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # 
        self.back_conv4 = nn.Sequential(nn.Conv2d(1024, feat_dim, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))

        self.back_conv3 = nn.Sequential(nn.Conv2d(1024, feat_dim, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))

        self.back_conv2 = nn.Sequential(nn.Conv2d(512, feat_dim, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))
        
        self.smooth_conv3 = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))

        self.smooth_conv2 = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(nn.Conv2d(feat_dim * 3, feat_dim, kernel_size=1),
                            nn.BatchNorm2d(feat_dim))                            

        self.attention = nn.Sequential(nn.Conv2d(feat_dim * 3, feat_dim, kernel_size=1),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(feat_dim, 1, kernel_size=1))
   
    def forward_train(self, x, mask, soft_train=True):        
        l1 = self.backbone[0](x)
        l2 = self.backbone[1](l1)
        l3 = self.backbone[2](l2)
        l4 = self.backbone[3](l3)

        p4 = self.back_conv4(l4)
        p3 = self._upsample_add(p4, self.back_conv3(l3))
        p3 = self.smooth_conv3(p3)
        p2 = self._upsample_add(p3, self.back_conv2(l2))
        p2 = self.smooth_conv2(p2)

        p = self._upsample_cat(p2, p3, p4)
        
        feat = self.conv(p)     # B x feat_dim x H x W   

        att = torch.sigmoid(self.attention(p))       
        
        mask = torch.nn.functional.interpolate(mask, size=att.shape[-2:], mode='nearest')
        mask = (mask > 0).float()
        
        if soft_train:
            feat = feat * mask.expand_as(feat) 
        else:
            feat = feat * att.expand_as(feat)  

        feat = feat.sum(dim=(2,3))
        feat = l2norm(feat)
        return feat, att, mask

    def forward(self, x, threshold=None):        
        l1 = self.backbone[0](x)
        l2 = self.backbone[1](l1)
        l3 = self.backbone[2](l2)
        l4 = self.backbone[3](l3)
        print(l1.shape, l2.shape, l3.shape, l4.shape)

        p4 = self.back_conv4(l4)
        p3 = self._upsample_add(p4, self.back_conv3(l3))
        p3 = self.smooth_conv3(p3)
        p2 = self._upsample_add(p3, self.back_conv2(l2))
        p2 = self.smooth_conv2(p2)

        p = self._upsample_cat(p2, p3, p4)
        
        feat = self.conv(p)     # B x feat_dim x H x W        
        att = torch.sigmoid(self.attention(p))
        
        if threshold:
            att = (att >= threshold).float()
        feat = feat * att.expand_as(feat)        
        feat = feat.sum(dim=(2,3))
        feat = l2norm(feat)
        return feat, att
    
    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

    def _upsample_cat(self, p2, p3, p4):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear')
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear')        
        return torch.cat([p2, p3, p4], dim=1)


class MultiScaleDense121Edge(nn.Module):
    # ref: https://github.com/WenmuZhou/DBNet.pytorch/blob/master/models/modules/segmentation_body.py    
    def __init__(self, feat_dim=256, pretrained=True, frozen=False):
        super(MultiScaleDense121Edge, self).__init__()         
        pretrain_net = densenet121(pretrained=pretrained).features     
        # init first conv
        conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            conv1.weight[:,:-1,:,:] = pretrain_net[0].weight.data 
            nn.init.xavier_uniform_(conv1.weight[:,-1,:,:])
        pretrain_net[0] = conv1

        self.backbone = nn.ModuleList()
        self.backbone.append(pretrain_net[0:5])
        self.backbone.append(pretrain_net[5:7])
        self.backbone.append(pretrain_net[7:9])
        self.backbone.append(pretrain_net[9:])

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # 
        self.back_conv4 = nn.Sequential(nn.Conv2d(1024, feat_dim, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))

        self.back_conv3 = nn.Sequential(nn.Conv2d(1024, feat_dim, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))

        self.back_conv2 = nn.Sequential(nn.Conv2d(512, feat_dim, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))
        
        self.smooth_conv3 = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))

        self.smooth_conv2 = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(nn.Conv2d(feat_dim * 3, feat_dim, kernel_size=1),
                            nn.BatchNorm2d(feat_dim))                            

        self.attention = nn.Sequential(nn.Conv2d(feat_dim * 3, feat_dim, kernel_size=1),
                            nn.BatchNorm2d(feat_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(feat_dim, 1, kernel_size=1))
   
    def forward_train(self, x, mask, soft_train=True):        
        l1 = self.backbone[0](x)
        l2 = self.backbone[1](l1)
        l3 = self.backbone[2](l2)
        l4 = self.backbone[3](l3)

        p4 = self.back_conv4(l4)
        p3 = self._upsample_add(p4, self.back_conv3(l3))
        p3 = self.smooth_conv3(p3)
        p2 = self._upsample_add(p3, self.back_conv2(l2))
        p2 = self.smooth_conv2(p2)

        p = self._upsample_cat(p2, p3, p4)
        
        feat = self.conv(p)     # B x feat_dim x H x W   

        att = torch.sigmoid(self.attention(p))       
        
        mask = torch.nn.functional.interpolate(mask, size=att.shape[-2:], mode='nearest')
        mask = (mask > 0).float()
        
        if soft_train:
            feat = feat * mask.expand_as(feat) 
        else:
            feat = feat * att.expand_as(feat)  

        feat = feat.sum(dim=(2,3))
        feat = l2norm(feat)
        return feat, att, mask

    def forward(self, x, threshold=None):        
        l1 = self.backbone[0](x)
        l2 = self.backbone[1](l1)
        l3 = self.backbone[2](l2)
        l4 = self.backbone[3](l3)

        p4 = self.back_conv4(l4)
        p3 = self._upsample_add(p4, self.back_conv3(l3))
        p3 = self.smooth_conv3(p3)
        p2 = self._upsample_add(p3, self.back_conv2(l2))
        p2 = self.smooth_conv2(p2)

        p = self._upsample_cat(p2, p3, p4)
        
        feat = self.conv(p)     # B x feat_dim x H x W        
        att = torch.sigmoid(self.attention(p))
        
        if threshold:
            att = (att >= threshold).float()
        feat = feat * att.expand_as(feat)        
        feat = feat.sum(dim=(2,3))
        feat = l2norm(feat)
        return feat, att
    
    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear') + y

    def _upsample_cat(self, p2, p3, p4):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear')
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear')        
        return torch.cat([p2, p3, p4], dim=1)


