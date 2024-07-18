#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/evaluator.py
# Project: /home/richard/sscp/utils
# Created Date: Tuesday, July 19th 2022, 2:47:37 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Dec 09 2022
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
#
# MIT License
#
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###
import os

import torch
import numpy as np
from tqdm import tqdm

from torch.distributed import ReduceOp
from loader.utils import Transform2Tensor


class Evaluator():

    def __init__(self, model, test_loader, criterion, device, cfg):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.test_loader = test_loader
        self.cfg = cfg

        self.to_tensor = Transform2Tensor(
            cfg.model.structure.spatial_edge_radius,
            cfg.model.structure.spatial_edge_min_distance,
            cfg.model.structure.sequential_edge_max_distance,
            cfg.model.structure.knn_k,
            cfg.model.structure.knn_min_distance,
            cfg.model.sequence.esm.token_arch,
        )

    def run(self):
        loss_list = []
        cl_acc_list = []

        self.model.eval()
        with torch.no_grad():
            loss = 0
            for step, batch_data in tqdm(
                    enumerate(self.test_loader),
                    desc="evaluating | loss: {}".format(loss)):

                struc_input, seq_input = self.to_tensor(batch_data)
                struc_input = struc_input.to(self.device)
                seq_input = seq_input.to(self.device)
                struc_rep, seq_rep = self.model(struc_input, seq_input)

                eval_loss, cl_acc = self.criterion(seq_rep, struc_rep,
                                                   self.cfg.loss)

                eval_loss = eval_loss.cpu().item()

                cl_acc_list.append(cl_acc)
                loss_list.append(eval_loss)

                if step >= self.cfg.inference.max_step:
                    break

        test_cl_acc = torch.tensor(np.mean(cl_acc_list)).to(self.device)
        test_loss = torch.tensor(np.mean(loss_list)).to(self.device)

        if self.cfg.mode.ddp:
            torch.distributed.barrier()
            torch.distributed.all_reduce(test_loss, op=ReduceOp.SUM)
            torch.distributed.all_reduce(test_cl_acc, op=ReduceOp.SUM)

            test_loss /= torch.distributed.get_world_size()
            test_cl_acc /= torch.distributed.get_world_size()

        # print("RANK: {}: loss {}".format(int(os.environ['RANK']), test_loss))
        # print("RANK: {}: ece {}".format(int(os.environ['RANK']), test_ece))
        # print("RANK: {}: acc {}".format(int(os.environ['RANK']), test_cl_acc))
        # print("RANK: {}: f1 {}".format(int(os.environ['RANK']), test_f1))

        return {
            "test_loss": test_loss.item(),
            "test_cl_acc": test_cl_acc.item(),
        }
