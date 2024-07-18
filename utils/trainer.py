from .std_logger import Logger
from .utils import is_parallel
from loader.utils import Transform2Tensor
import matplotlib.pyplot as plt

import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
import random
import mlflow
from tqdm import tqdm
import shutil
import os


class Trainer(object):

    def __init__(self, net, AE_2D_3D, AE_3D_2D, criterion, dataloaders, optimizer, scheduler,
                 device, global_rank, world_size, cfg):

        self.net = net
        self.AE_2D_3D = AE_2D_3D
        self.AE_3D_2D = AE_3D_2D
        self.device = device
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = cfg.train.num_epoch
        self.batch_size = cfg.train.batch_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.cfg = cfg
        self.global_train_step = 0
        self.global_valid_step = 0

        ## save checkpoint
        self.root_level_dir = os.path.dirname(
            os.path.join(os.path.dirname(os.path.abspath(__file__))))

        ## amp
        self.scaler = None

        self.to_tensor = Transform2Tensor(
            cfg.model.structure.spatial_edge_radius,
            cfg.model.structure.spatial_edge_min_distance,
            cfg.model.structure.sequential_edge_max_distance,
            cfg.model.structure.knn_k,
            cfg.model.structure.knn_min_distance,
            cfg.model.sequence.esm.token_arch,
        )

    def evaluate(self, split):
        loss_list = []
        cl_acc_list = []

        self.net.eval()
        with torch.no_grad():
            eval_loss = cl_acc = 0

            for step, batch_data in tqdm(
                    enumerate(self.dataloaders[split]),
                    desc="evaluating | loss: {}, acc: {}".format(
                        eval_loss, cl_acc)):

                struc_input, seq_input = self.to_tensor(batch_data)
                struc_input = struc_input.to(self.device)
                seq_input = seq_input.to(self.device)

                struc_rep, seq_rep = self.net(struc_input, seq_input)
                AE_loss_1 = self.AE_2D_3D(seq_rep, struc_rep)
                AE_loss_2 = self.AE_3D_2D(struc_rep, seq_rep)
                eval_loss, cl_acc = self.criterion(seq_rep, struc_rep, self.cfg.loss)
                AE_loss = (AE_loss_1 + AE_loss_2) / 2
                eval_loss += 0.2 * AE_loss
                eval_loss = eval_loss.detach().cpu().item()

                cl_acc_list.append(cl_acc)
                loss_list.append(eval_loss)

                if self.cfg.other.debug and step >= self.cfg.other.debug_step:
                    break

        return {
            "{}_loss".format(split): np.nanmean(loss_list),
            "{}_cl_acc".format(split): np.mean(cl_acc_list),
        }

    def run(self):

        loss_list = []
        acc_list = []
        if self.cfg.train.amp:
            self.scaler = GradScaler(init_scale=2**16,
                                     growth_factor=2,
                                     backoff_factor=0.5,
                                     growth_interval=2000,
                                     enabled=True)

        for epoch in range(self.num_epoch):

            self.net.train()
            acc_cum = 0
            loss_cum = 0
            count = 0
            # self.dataloaders["train"].sampler.set_epoch(epoch)

            for _, batch_data in enumerate(self.dataloaders["train"]):

                # print(batch_data['sequence'])
                # len_ll = [len(i) for i in batch_data['sequence']]
                # print(len_ll)

                struc_input, seq_input = self.to_tensor(batch_data)
                struc_input = struc_input.to(self.device)
                seq_input = seq_input.to(self.device)

                # forward
                if self.cfg.train.amp:
                    with autocast():
                        struc_rep, seq_rep = self.net(struc_input, seq_input)
                        AE_loss_1 = self.AE_2D_3D(seq_rep, struc_rep)
                        AE_loss_2 = self.AE_3D_2D(struc_rep, seq_rep)
                        train_loss, cl_acc = self.criterion(seq_rep, struc_rep, self.cfg.loss)
                        AE_loss = (AE_loss_1 + AE_loss_2) / 2
                        train_loss += 0.2 * AE_loss

                else:
                    struc_rep, seq_rep = self.net(struc_input, seq_input)

                    AE_loss_1 = self.AE_2D_3D(seq_rep, struc_rep)
                    AE_loss_2 = self.AE_3D_2D(struc_rep, seq_rep)
                    train_loss, cl_acc = self.criterion(seq_rep, struc_rep, self.cfg.loss)
                    AE_loss = (AE_loss_1 + AE_loss_2) / 2
                    train_loss += 0.2 * AE_loss

                acc_cum += cl_acc
                loss_cum += train_loss
                count += 1

                # backward
                if self.cfg.train.gradient_accumulation_steps > 1:
                    train_loss /= self.cfg.train.gradient_accumulation_steps

                if (self.global_train_step +
                        1) % self.cfg.train.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad()
                    if self.cfg.train.amp:
                        self.scaler.scale(train_loss).backward()
                    else:
                        train_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.train.max_grad_norm)

                    torch.nn.utils.clip_grad_norm_(
                        self.AE_2D_3D.parameters(), self.cfg.train.max_grad_norm)

                    torch.nn.utils.clip_grad_norm_(
                        self.AE_3D_2D.parameters(), self.cfg.train.max_grad_norm)

                    if self.cfg.train.amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()

                ## logging
                if (self.global_train_step +
                        1) % self.cfg.logger.log_per_steps == 0:

                    cur_lr = self.scheduler.optimizer.state_dict(
                    )['param_groups'][0]['lr']
                    print("train | epoch: {:d}  step: {:d} | loss: {:.4f}".
                            format(epoch, self.global_train_step,
                                   train_loss.item()))

                    if self.global_rank == 0 and self.cfg.logger.log:
                        mlflow.log_metric("train/vae_loss",
                                          0.2 * AE_loss.item(),
                                          step=self.global_train_step)

                        mlflow.log_metric("train/loss",
                                          train_loss.item(),
                                          step=self.global_train_step)
                        mlflow.log_metric("train/cl_acc",
                                          cl_acc,
                                          step=self.global_train_step)
                        mlflow.log_metric("lr",
                                          cur_lr,
                                          step=self.global_train_step)
                        Logger.info("lr: {:.8f}".format(cur_lr))

                        Logger.info(
                            "train | epoch: {:d}  step: {:d} | vae loss: {:.4f} | loss: {:.4f}".
                            format(epoch, self.global_train_step, 0.2 * AE_loss.item(), train_loss.item()))

                ### evaluating
                if (self.global_train_step + 1) % self.cfg.train.eval_per_steps == 0:

                    if self.global_rank == 0:
                        valid_metrics = self.evaluate("valid")

                        Logger.info(
                            "valid | epoch: {:d} | step: {:d} | loss: {:.4f} | cl_acc: {:.4f}"
                            .format(epoch, self.global_valid_step,
                                    valid_metrics["valid_loss"],
                                    valid_metrics["valid_cl_acc"]))

                        if self.cfg.logger.log:
                            for metric_name, metric_v in valid_metrics.items():
                                mlflow.log_metric(
                                    "valid/{}".format(metric_name),
                                    metric_v,
                                    step=self.global_valid_step)

                            self.model_path = Path(self.root_level_dir + "/checkpoints/model_step_{}".format(self.global_valid_step))

                            if self.model_path.exists():
                                shutil.rmtree(self.model_path)

                            mlflow.pytorch.save_model(
                                (self.net.module
                                 if is_parallel(self.net) else self.net),
                                self.model_path,
                                code_paths=[
                                    os.path.join(self.root_level_dir, "model")
                                ])
                        else:
                            self.model_path = Path(self.root_level_dir + "/checkpoints/{}/model_step_{}".format(
                                self.cfg.data.data_name,self.global_valid_step))

                            if not os.path.exists(self.model_path):
                                os.makedirs(self.model_path)
                            self.model_file = os.path.join(self.model_path, 'pretrain_model.pth')
                            torch.save(self.net.module,self.model_file)

                        self.global_valid_step += 1
                        self.net.train()

                self.global_train_step += 1

                if self.cfg.other.debug and self.global_train_step >= self.cfg.other.debug_step:
                    break

            loss_list.append(loss_cum)
            acc_list.append(acc_cum)

        return loss_list, acc_list