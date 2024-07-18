import os
import sys

sys.path.append("..")

import torch
import torch.nn as nn
from model.structure_view import StructureView
from model.sequence_view import SequenceView, Pep_SequenceView
from model.str_seq_contrastive import MultiviewContrast
from torchdrug.layers import geometry
import mlflow


class SeqStrMultiView(nn.Module):

    def __init__(self, seq_model_path, seq_model_name, struc_model_path,
                 input_dim, hidden_dims,
                 num_relation, edge_input_dim, num_angle_bin, batch_norm,
                 concat_hidden, short_cut, readout, mask_rate, out_dim):

        super(SeqStrMultiView, self).__init__()
        if 'pep' in seq_model_path:
            seq_model = Pep_SequenceView(seq_model_path, seq_model_name)
        else:
            seq_model = SequenceView(seq_model_path, seq_model_name)
        struc_model = StructureView(input_dim=input_dim,
                                    hidden_dims=hidden_dims,
                                    num_relation=num_relation,
                                    edge_input_dim=edge_input_dim,
                                    num_angle_bin=num_angle_bin,
                                    batch_norm=batch_norm,
                                    concat_hidden=concat_hidden,
                                    short_cut=short_cut,
                                    readout=readout,
                                    weight_path=struc_model_path)

        self.model = MultiviewContrast(
            seq_model,
            struc_model,
            noise_funcs=[
                geometry.IdentityNode(),
                geometry.RandomEdgeMask(mask_rate=mask_rate)
            ],
            crop_funcs=[],
            out_dim=out_dim)

    def forward(self, struc_input, seq_input):
        repr_dict = self.model(struc_input, seq_input)
        struc_rep = repr_dict["struc_rep"]
        seq_rep = repr_dict["seq_rep"]

        return struc_rep, seq_rep
