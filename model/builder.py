#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/sscp/model/builder.py
# Project: /home/richard/sscp/model
# Created Date: Friday, December 9th 2022, 6:36:08 am
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
# Copyright (c) 2022 Ruochi Zhang
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
import sys
import mlflow
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.core import SeqStrMultiView, SequenceView, Pep_SequenceView


def model_factory(orig_cwd, model_cfg, device):

    model = SeqStrMultiView(
        seq_model_path=os.path.join(orig_cwd, model_cfg.sequence.esm.model_path),
        seq_model_name=model_cfg.sequence.esm.model_name,
        struc_model_path=model_cfg.structure.model_path,
        input_dim=model_cfg.contrastive.input_dim,
        hidden_dims=model_cfg.contrastive.hidden_dims,
        num_relation=model_cfg.contrastive.num_relation,
        edge_input_dim=model_cfg.contrastive.edge_input_dim,
        num_angle_bin=model_cfg.contrastive.num_angle_bin,
        batch_norm=model_cfg.contrastive.batch_norm,
        concat_hidden=model_cfg.contrastive.concat_hidden,
        short_cut=model_cfg.contrastive.short_cut,
        readout=model_cfg.contrastive.readout,
        mask_rate=model_cfg.contrastive.mask_rate,
        out_dim=model_cfg.contrastive.out_dim)
    
    model = model.to(device)
    return model

def load_pretrain_model(orig_cwd,cfg,device):

    model_path = cfg.model.pretrain.model_path
    model_path = os.path.join(orig_cwd, model_path)
    sys.path.append(os.path.join(model_path, "code"))
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    model = model.to(device)
    return model