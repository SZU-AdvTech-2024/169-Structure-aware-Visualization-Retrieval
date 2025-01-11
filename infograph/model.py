import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ModuleList, Linear, GRU, ReLU, BatchNorm1d

from dgl.nn import GINConv, NNConv, Set2Set
from dgl.nn.pytorch.glob import SumPooling, AvgPooling

from utils import global_global_loss_, local_global_loss_


''' Feedforward neural network'''
class FeedforwardNetwork(nn.Module):

    '''
    3-layer feed-forward neural networks with jumping connections
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.

    Functions
    -----------
    forward(feat):
        feat: Tensor
            [N * D], input features
    '''

    def __init__(self, in_dim, hid_dim):
        super(FeedforwardNetwork, self).__init__()

        self.block = Sequential(Linear(in_dim, hid_dim),
                                ReLU(),
                                Linear(hid_dim, hid_dim),
                                ReLU(),
                                Linear(hid_dim, hid_dim),
                                ReLU()
                                )

        self.jump_con = Linear(in_dim, hid_dim)

    def forward(self, feat):
        block_out = self.block(feat)
        jump_out = self.jump_con(feat)

        out = block_out + jump_out

        return out


''' Unsupervised Setting '''

class GINEncoder(nn.Module):
    '''
    Encoder based on dgl.nn.GINConv &  dgl.nn.SumPooling
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    n_layer: 
        Number of GIN layers.

    Functions
    -----------
    forward(graph, feat):
        graph: DGLGraph
        feat: Tensor
            [N * D], node features
    '''

    def __init__(self, in_dim, hid_dim, n_layer):
        super(GINEncoder, self).__init__()

        self.n_layer = n_layer

        self.convs = ModuleList()
        self.bns = ModuleList()

        for i in range(n_layer):
            if i == 0:
                n_in = in_dim
            else:
                n_in = hid_dim
            n_out = hid_dim
            block = Sequential(Linear(n_in, n_out),
                               ReLU(),
                               Linear(hid_dim, hid_dim)
                               )

            conv = GINConv(apply_func = block, aggregator_type = 'mean')
            bn = BatchNorm1d(hid_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        # sum pooling
        self.pool = AvgPooling()

    def forward(self, graph, feat):

        xs = []
        x = feat
        for i in range(self.n_layer):
            x = F.relu(self.convs[i](graph, x))
            x = self.bns[i](x)
            xs.append(x)

        local_emb = th.cat(xs, 1)                    # patch-level embedding
        global_emb = self.pool(graph, local_emb)     # graph-level embedding

        return global_emb, local_emb


class InfoGraph(nn.Module):
    r"""
        InfoGraph model for unsupervised setting

    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    n_layer: int   
        Number of the GNN encoder layers.

    Functions
    -----------
    forward(graph):
        graph: DGLGraph

    """

    def __init__(self, in_dim, hid_dim, n_layer):
        super(InfoGraph, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim

        self.n_layer = n_layer
        embedding_dim = hid_dim * n_layer

        self.encoder = GINEncoder(in_dim, hid_dim, n_layer)

        self.local_d = FeedforwardNetwork(embedding_dim, embedding_dim)   # local discriminator (node-level)
        self.global_d = FeedforwardNetwork(embedding_dim, embedding_dim)  # global discriminator (graph-level)

    def get_embedding(self, graph, feat):
        # get_embedding function for evaluation the learned embeddings

        with th.no_grad():
            global_emb, _ = self.encoder(graph, feat)

        return global_emb

    def forward(self, graph, feat, graph_id):

        global_emb, local_emb = self.encoder(graph, feat)

        global_h = self.global_d(global_emb)    # global hidden representation
        local_h = self.local_d(local_emb)       # local hidden representation

        loss = local_global_loss_(local_h, global_h, graph_id)

        return loss


