#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoesm/train.py
# Project: /home/richard/sscp
# Created Date: Saturday, July 16th 2022, 7:14:39 pm
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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import hydra
import torch
import warnings
warnings.filterwarnings('ignore')

from utils.loss import dual_CL
from loader.utils import make_loaders
from utils.utils import fix_random_seed
from utils.std_logger import Logger
from utils.utils import get_device
from utils.trainer import Trainer
from utils.distribution import setup_multinodes, cleanup_multinodes
from model.builder import model_factory
from model.auto_encoder import VariationalAutoEncoder
from utils.utils import plot_training_trace

import mlflow
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):

    orig_cwd = hydra.utils.get_original_cwd()

    global_rank = 0
    local_rank = 0
    world_size = 0

    if cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    if cfg.mode.ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        os.environ['NCCL_DEBUG'] = 'INFO'
        os.environ['NCCL_SHM_DISABLE'] = '1'
        random_seed = cfg.train.random_seed + local_rank
    else:
        random_seed = cfg.train.random_seed

    if cfg.logger.log and global_rank == 0:
        # log hyper-parameters
        for p, v in cfg.data.items():
            mlflow.log_param(p, v)

        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        for p, v in cfg.model.items():
            mlflow.log_param(p, v)

    fix_random_seed(random_seed, cuda_deterministic=True)

    if cfg.mode.ddp:
        setup_multinodes(local_rank, world_size)

    if cfg.mode.ddp:
        device = torch.device("cuda", local_rank)
    else:
        device = get_device(cfg)

    if global_rank == 0:
        Logger.info("Using device {}......".format(device))

    #-------------------- load dataset --------------------
    dataloaders = make_loaders(orig_cwd, local_rank, cfg=cfg)

    # -------------------- load model --------------------

    model = model_factory(orig_cwd, cfg.model, device)
    AE_2D_3D_model = VariationalAutoEncoder(emb_dim=cfg.model.contrastive.out_dim, loss='l2', detach_target=True, beta=0.5).to(device)
    AE_3D_2D_model = VariationalAutoEncoder(emb_dim=cfg.model.contrastive.out_dim, loss='l2', detach_target=True, beta=0.5).to(device)

    if cfg.mode.ddp:
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True)

    # -------------------- load optimizer --------------------
    lr_list = [{'params': model.model.model_seq.parameters(), 'lr': cfg.train.lr_sequence_model},
               {'params': model.model.model_struc.parameters(), 'lr': cfg.train.lr_structure_model},
               {'params': model.model.mlp_seq.parameters(), 'lr': cfg.train.lr_sequence_model},
               {'params': model.model.mlp_struc.parameters(), 'lr': cfg.train.lr_structure_model},
               {'params': AE_2D_3D_model.parameters(), 'lr': cfg.train.lr_sequence_model},
               {'params': AE_3D_2D_model.parameters(), 'lr': cfg.train.lr_structure_model}]

    optimizer = torch.optim.Adam(lr_list, weight_decay=cfg.train.weight_decay)

    class makeschuler():
        def __init__(self, optimizer):
            self.optimizer = optimizer
        def step(self):
            return

    scheduler = makeschuler(optimizer)

    # -------------------- load loss function --------------------
    criterion = dual_CL

    trainer = Trainer(model, AE_2D_3D_model, AE_3D_2D_model, criterion, dataloaders, optimizer, scheduler,
                      device, global_rank, world_size, cfg)

    if global_rank == 0:
        Logger.info("start training......")

    loss_list, acc_list = trainer.run()

    if cfg.other.plot:
        plot_training_trace(loss_list, acc_list)

    if global_rank == 0:
        Logger.info("finished training......")

    if global_rank == 0:
        Logger.info("loading best weights......")

    if cfg.mode.ddp:
        cleanup_multinodes()


if __name__ == "__main__":
    main()