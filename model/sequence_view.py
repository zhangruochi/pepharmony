#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /home/richard/sscp/model/sequence_view.py
# Project: /home/richard/sscp/model
# Created Date: Friday, December 9th 2022, 5:51:36 am
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

sys.path.append("..")

import torch
import torch.nn as nn
from model import esm
from model.helpers import Permute, Squeeze
from torchdrug.layers import functional
from torchdrug import layers
import mlflow



class SequenceView(nn.Module):

    def __init__(
        self,
        model_path,
        model_name,
        readout='sum'
    ):
        super(SequenceView, self).__init__()

        model_data = torch.load(os.path.join(model_path,
                                             "{}.pt".format(model_name)),
                                map_location="cpu")
        regression_data = torch.load(
            os.path.join(model_path,
                         '{}-contact-regression.pt'.format(model_name)))
        self.feature_extractor, self.alphabet = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data)

        self.output_dim = self.feature_extractor.embed_dim

        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

        self.pool = nn.Sequential(
            Permute(), nn.AdaptiveAvgPool1d(1),
            Squeeze(dim=-1))


    def forward(self, graph, input, all_loss=None, metric=None):

        amino_acid_rep = self.feature_extractor(input.long(), repr_layers=[12], return_contacts=False)["representations"][12]
        graph_feature = self.pool(amino_acid_rep)

        return {
            "protein_feature": graph_feature,
            "residue_feature": amino_acid_rep
        }

def load_from_mlflow(model_path):
    """ load pre-trained peptide model from mlflow
    """
    # model_path = os.path.join(root_dir, model_path)
    
    sys.path.append(os.path.join(model_path, "code"))
    # sys.path.append('/home/xiuyuting')  # pretrained model not correctly load, sys.path issue
    model_peptide = mlflow.pytorch.load_model(model_path, map_location="cpu")
    return model_peptide

class Pep_SequenceView(nn.Module):

    def __init__(
        self,
        model_path,
        model_name,
        readout='sum'
    ):
        super(Pep_SequenceView, self).__init__()

        self.feature_extractor = load_from_mlflow(model_path='/mnt/share/pretrained_models/pep_esm_t12/model_step_2246_ece_10.999')

        self.output_dim = self.feature_extractor.embed_dim
        
        if readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)
        
        self.pool = nn.Sequential(
            Permute(), nn.AdaptiveAvgPool1d(1),
            Squeeze(dim=-1))


    def forward(self, graph, input, all_loss=None, metric=None):

        amino_acid_rep = self.feature_extractor(input.long(), repr_layers=[12], return_contacts=False)["representations"][12]
        graph_feature = self.pool(amino_acid_rep)

        return {
            "protein_feature": graph_feature,
            "residue_feature": amino_acid_rep
        }
