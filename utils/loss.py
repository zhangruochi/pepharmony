import logging
import random
from math import sqrt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import seaborn as sns
import matplotlib.pyplot as plt
import time


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, loss_config):
    if loss_config.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if loss_config.CL_similarity_metric == 'InfoNCE_dot_prod':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, loss_config.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1
        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    elif loss_config.CL_similarity_metric == 'EBM_dot_prod':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([
            Y[cycle_index(len(Y), i + 1)]
            for i in range(loss_config.CL_neg_samples)
        ],
                          dim=0)
        neg_X = X.repeat((loss_config.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / loss_config.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / loss_config.T

        loss_pos = criterion(pred_pos,
                             torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg,
                             torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + loss_config.CL_neg_samples * loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() +
                  torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    else:
        raise Exception

    return CL_loss, CL_acc


def dual_CL(X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2


def do_GraphCL(batch1, batch2, molecule_model_2D, projection_head,
               molecule_readout_func):
    x1 = molecule_model_2D(batch1.x, batch1.edge_index, batch1.edge_attr)
    x1 = molecule_readout_func(x1, batch1.batch)
    x1 = projection_head(x1)

    x2 = molecule_model_2D(batch2.x, batch2.edge_index, batch2.edge_attr)
    x2 = molecule_readout_func(x2, batch2.batch)
    x2 = projection_head(x2)

    T = 0.1
    batch, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum(
        'i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch), range(batch)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = -torch.log(loss).mean()
    return loss


def do_GraphCLv2(batch1, batch2, n_aug1, n_aug2, molecule_model_2D,
                 projection_head, molecule_readout_func):
    x1 = molecule_model_2D(batch1.x, batch1.edge_index, batch1.edge_attr)
    x1 = molecule_readout_func(x1, batch1.batch)
    x1 = projection_head[n_aug1](x1)

    x2 = molecule_model_2D(batch2.x, batch2.edge_index, batch2.edge_attr)
    x2 = molecule_readout_func(x2, batch2.batch)
    x2 = projection_head[n_aug2](x2)

    T = 0.1
    batch, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum(
        'i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch), range(batch)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = -torch.log(loss).mean()
    return loss