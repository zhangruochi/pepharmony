from torchdrug import transforms
from torchdrug.layers import functional
from torchdrug.layers import geometry
from torchdrug import core, data
from torchdrug.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch import nn
import os
from utils.std_logger import Logger
from loader.dataset import EnzymeCommissionDataset, PDBPeptide, AlphaFoldDB, CustomSubset
from model.esm.data import Alphabet
from copy import deepcopy
import numpy as np

class GraphConstruction(nn.Module, core.Configurable):
    """
    Construct a new graph from an existing graph.

    See `torchdrug.layers.geometry` for a full list of available node and edge layers.

    Parameters:
        node_layers (list of nn.Module, optional): modules to construct nodes of the new graph
        edge_layers (list of nn.Module, optional): modules to construct edges of the new graph
        edge_feature (str, optional): edge features in the new graph.
            Available features are ``residue_type``, ``gearnet``.

            1. For ``residue_type``, the feature of the edge :math:`e_{ij}` between residue :math:`i` and residue
                :math:`j` is the concatenation ``[residue_type(i), residue_type(j)]``.
            2. For ``gearnet``, the feature of the edge :math:`e_{ij}` between residue :math:`i` and residue :math:`j`
                is the concatenation ``[residue_type(i), residue_type(j), edge_type(e_ij),
                sequential_distance(i,j), spatial_distance(i,j)]``.
    """

    max_seq_dist = 10

    def __init__(self, node_layers=None, edge_layers=None, edge_feature="residue_type"):
        super(GraphConstruction, self).__init__()
        if node_layers is None:
            self.node_layers = nn.ModuleList()
        else:
            self.node_layers = nn.ModuleList(node_layers)
        if edge_layers is None:
            edge_layers = nn.ModuleList()
        else:
            edge_layers = nn.ModuleList(edge_layers)
        self.edge_layers = edge_layers
        self.edge_feature = edge_feature

    def edge_residue_type(self, graph, edge_list):
        node_in, node_out, _ = edge_list.t()
        residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]

        return torch.cat([
            functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(out_residue_type, len(data.Protein.residue2id))
        ], dim=-1)

    def edge_gearnet(self, graph, edge_list, num_relation):
        node_in, node_out, r = edge_list.t()
        residue_in, residue_out = graph.atom2residue[node_in], graph.atom2residue[node_out]
        in_residue_type = graph.residue_type[residue_in]
        out_residue_type = graph.residue_type[residue_out]
        sequential_dist = torch.abs(residue_in - residue_out)
        spatial_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)

        # print(graph.node_position[node_in].size())
        # print(graph.node_position[node_out].size())
        # print(functional.one_hot(r, num_relation).size())
        # print(aa.size())
        # print(spatial_dist.unsqueeze(-1))

        return torch.cat([
            graph.node_position[node_in],
            graph.node_position[node_out],
            # functional.one_hot(in_residue_type, len(data.Protein.residue2id)),
            # functional.one_hot(out_residue_type, len(data.Protein.residue2id)),
            functional.one_hot(r, num_relation),
            functional.one_hot(sequential_dist.clamp(max=self.max_seq_dist), self.max_seq_dist + 1),
            spatial_dist.unsqueeze(-1)
        ], dim=-1)

    def apply_node_layer(self, graph):
        for layer in self.node_layers:
            graph = layer(graph)
        return graph

    def apply_edge_layer(self, graph):
        if not self.edge_layers:
            return graph

        edge_list = []
        num_edges = []
        num_relations = []
        for layer in self.edge_layers:
            edges, num_relation = layer(graph)
            edge_list.append(edges)
            num_edges.append(len(edges))
            num_relations.append(num_relation)

        edge_list = torch.cat(edge_list)
        num_edges = torch.tensor(num_edges, device=graph.device)
        num_relations = torch.tensor(num_relations, device=graph.device)
        num_relation = num_relations.sum()
        offsets = (num_relations.cumsum(0) - num_relations).repeat_interleave(num_edges)
        edge_list[:, 2] += offsets

        # reorder edges into a valid PackedGraph
        node_in = edge_list[:, 0]
        edge2graph = graph.node2graph[node_in]
        order = edge2graph.argsort()
        edge_list = edge_list[order]
        num_edges = edge2graph.bincount(minlength=graph.batch_size)
        offsets = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(num_edges)

        if self.edge_feature == "residue_type":
            edge_feature = self.edge_residue_type(graph, edge_list)
        elif self.edge_feature == "gearnet":
            edge_feature = self.edge_gearnet(graph, edge_list, num_relation)

        else:
            raise ValueError("Unknown edge feature `%s`" % self.edge_feature)
        data_dict, meta_dict = graph.data_by_meta(include=("node", "residue", "node reference", "residue reference"))

        if isinstance(graph, data.PackedProtein):
            data_dict["num_residues"] = graph.num_residues
        if isinstance(graph, data.PackedMolecule):
            data_dict["bond_type"] = torch.zeros_like(edge_list[:, 2])
        return type(graph)(edge_list, num_nodes=graph.num_nodes, num_edges=num_edges, num_relation=num_relation,
                           view=graph.view, offsets=offsets, edge_feature=edge_feature,
                           meta_dict=meta_dict, **data_dict)

    def forward(self, graph):
        """
        Generate a new graph based on the input graph and pre-defined node and edge layers.

        Parameters:
            graph (Graph): :math:`n` graph(s)

        Returns:
            graph (Graph): new graph(s)
        """
        graph = self.apply_node_layer(graph)
        graph = self.apply_edge_layer(graph)
        return graph


class Transform2Tensor():

    def __init__(
        self,
        spatial_edge_radius,
        spatial_edge_min_distance,
        sequential_edge_max_distance,
        knn_k,
        knn_min_distance,
        token_arch,
    ):

        self.graph_construction_model = GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[
                geometry.SpatialEdge(radius=spatial_edge_radius,
                                     min_distance=spatial_edge_min_distance),
                geometry.KNNEdge(k=knn_k, min_distance=knn_min_distance),
                geometry.SequentialEdge(
                    max_distance=sequential_edge_max_distance)
            ],
            edge_feature="gearnet")

        self.tokenizer = Alphabet.from_architecture(
            token_arch).get_batch_converter()

    def __call__(self, batch):

        structure_input = self.graph_construction_model(batch['graph'])

        # Get augmented views
        with structure_input.residue():
            structure_input.input = structure_input.node_feature.float()

        sequence_input = [("protein_{}".format(_), seq)
                          for _, seq in enumerate(batch["sequence"])]

        batch_labels, batch_strs, batch_tokens = self.tokenizer(sequence_input)

        return structure_input, batch_tokens

def generate_dataloader(dataset, ddp, batch_size, num_workers, pin_memory):

    if ddp:
        sampler = DistributedSampler(dataset, rank=torch.distributed.get_rank())
        shuffle = None
    else:
        sampler = None
        shuffle = False   # Have to be false here to give the control of sequence length in per batch

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=shuffle,
                            drop_last=True,
                            sampler=sampler)
    return dataloader

def get_transforms(transform_config):

    truncuate_transform = transforms.TruncateProtein(max_length=transform_config.truncate_length, random=False)
    protein_view_transform = transforms.ProteinView(view='residue')

    return transforms.Compose([truncuate_transform, protein_view_transform])

def make_loaders(orig_cwd, local_rank, cfg):

    if cfg.data.type == "toy":

        dataset = EnzymeCommissionDataset(
            local_rank=local_rank,
            path=os.path.join(orig_cwd, cfg.data.path),
            test_cutoff=cfg.data.test_cutoff,
            transform=get_transforms(cfg.model.structure.transform),
            atom_feature=None,
            bond_feature=None)
        train_set, valid_set, test_set = dataset.split()

    elif cfg.data.type == "pdb":

        dataset = PDBPeptide(
            local_rank=local_rank,
            path=cfg.data.path,
            split_indices=None,
            transform=get_transforms(cfg.model.structure.transform),
            atom_feature=None,
            bond_feature=None)

        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = int(len(dataset) * 0.1)

        origin_data = deepcopy(dataset)
        idx = np.random.choice(range(len(dataset)), len(dataset), replace=False, p=None)
        train_set = CustomSubset(origin_data, idx[:train_set_size], get_transforms(cfg.model.structure.transform),
                                 local_rank=local_rank)
        valid_set = CustomSubset(origin_data, idx[train_set_size: train_set_size + valid_set_size],
                                 get_transforms(cfg.model.structure.transform), local_rank=local_rank)
        test_set = CustomSubset(origin_data, idx[train_set_size + valid_set_size:],
                                get_transforms(cfg.model.structure.transform), local_rank=local_rank)

    elif cfg.data.type == "alphafold":
        dataset = AlphaFoldDB(
            local_rank=local_rank,
            path=cfg.data.path,
            pkl_path=os.path.join(orig_cwd, cfg.data.pkl_path),
            mini_dataset_len=cfg.data.mini_data_len,
            transform=get_transforms(cfg.model.structure.transform),
            atom_feature=None,
            bond_feature=None)

        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = int(len(dataset) * 0.1)
        # train_set, valid_set, test_set = random_split(dataset, [train_set_size, valid_set_size, test_set_size],
        #                                               generator=torch.Generator().manual_seed(0))

        origin_data = deepcopy(dataset)
        idx = np.random.choice(range(len(dataset)), len(dataset), replace=False, p=None)
        train_set = CustomSubset(origin_data, idx[:train_set_size],
                                 get_transforms(cfg.model.structure.transform), local_rank=local_rank)
        valid_set = CustomSubset(origin_data, idx[train_set_size: train_set_size+valid_set_size],
                                 get_transforms(cfg.model.structure.transform), local_rank=local_rank)
        test_set = CustomSubset(origin_data, idx[train_set_size+valid_set_size:],
                                get_transforms(cfg.model.structure.transform), local_rank=local_rank)

    else:
        raise "unknown dataset"

    Logger.info('start loading features.')
    print("train samples: %d, valid samples: %d, test samples: %d" %
          (len(train_set), len(valid_set), len(test_set)))

    Logger.info("train dataloader......")
    train_loader = generate_dataloader(train_set, cfg.mode.ddp,
                                       cfg.train.batch_size,
                                       cfg.train.num_workers,
                                       cfg.train.pin_memory)
    Logger.info("valid dataloader......")
    valid_loader = generate_dataloader(valid_set, cfg.mode.ddp,
                                       cfg.train.batch_size,
                                       cfg.train.num_workers,
                                       cfg.train.pin_memory)
    Logger.info("test dataloader......")
    test_loader = generate_dataloader(test_set, cfg.mode.ddp,
                                      cfg.train.batch_size,
                                      cfg.train.num_workers,
                                      cfg.train.pin_memory)

    dataset_loader = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }

    return dataset_loader