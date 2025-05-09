import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn import Linear
from torch.nn import LayerNorm, BatchNorm1d, Identity
from torch_geometric.nn import GATConv, GCNConv


class GNNConv(nn.Module):
    def __init__(self, conv_name, in_channels, out_channels, norm,
                 self_loop=True, n_heads=[1, 1], iscat=[False, False], dropout_att=0.):
        super(GNNConv, self).__init__()
        self.name = conv_name

        if conv_name == 'gcn_conv':
            self.conv  = GCNConv(in_channels, out_channels, add_self_loops=self_loop)
            self.conv_ = self.conv.lin

        elif conv_name == 'gat_conv':
            if iscat[0]: # if previous gatconv's cat is True
                in_channels = in_channels * n_heads[0]
            self.conv  = GATConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 heads=n_heads[1],
                                 concat=iscat[1],
                                 dropout=dropout_att,
                                 add_self_loops=self_loop)
            self.conv_ = self.conv.lin_src
            if iscat[1]: # if this gatconv's cat is True
                out_channels = out_channels * n_heads[1]

        if norm == 'LayerNorm':
            self.norm, self.norm_ = LayerNorm(out_channels), LayerNorm(out_channels)
        elif norm == 'BatchNorm1d':
            self.norm, self.norm_ = BatchNorm1d(out_channels), BatchNorm1d(out_channels)
        else:
            self.norm, self.norm_ = Identity(), Identity()


    def forward(self, xs, edge_index):
        if isinstance(xs, list): # if xs is [x, x_], we use twin-gnn
            x = self.conv(xs[0], edge_index)
            x_ = self.conv_(xs[1])
            return self.norm(x), self.norm_(x_)

        else: # if xs is x, we use sigle-gnn
            x = xs
            x = self.conv(x, edge_index)
            return self.norm(x)
        
class SkipConnection(nn.Module):
    def __init__(self, skip_connection, in_channels, out_channels):
        super(SkipConnection, self).__init__()
        self.skip_connection = skip_connection

        if in_channels == out_channels:
            self.transformer = Identity()
        else:
            self.transformer = Linear(in_channels, out_channels)

        if self.skip_connection == 'highway':
            self.gate_linear = Linear(out_channels, out_channels)

    def forward(self, h_x):
        if isinstance(h_x, list): # if h_x_ is [(h,x), (h_,x_)], we use twin-skip
            h,  x  = h_x[0]
            h_, x_ = h_x[1]
    
            if self.skip_connection == 'vanilla':
                return h, h_

            else: # if use any skip_connection
                x  = self.transformer(x) # in_channels >> out_channels
                x_ = self.transformer(x_)

                if self.skip_connection == 'res':
                    return h + x, h_ + x_

                elif self.skip_connection == 'dense':
                    return torch.cat([h, x], dim=-1), torch.cat([h_, x_], dim=-1)

                elif self.skip_connection == 'highway':
                    gating_weights = torch.sigmoid(self.gate_linear(x))
                    ones = torch.ones_like(gating_weights)
                    return h*gating_weights + x*(ones-gating_weights), \
                           h_*gating_weights + x_*(ones-gating_weights)

        else: # if h_x_ is (h,x), we use single-skip
            h,  x  = h_x

            if self.skip_connection == 'vanilla':
                return h

            else: # if use any skip_connection
                x  = self.transformer(x) # in_channels >> out_channels

                if self.skip_connection == 'res':
                    return h + x

                elif self.skip_connection == 'dense':
                    return torch.cat([h, x], dim=-1)

                elif self.skip_connection == 'highway':
                    gating_weights = torch.sigmoid(self.gate_linear(x))
                    ones = torch.ones_like(gating_weights)
                    return h*gating_weights + x*(ones-gating_weights)




def orthonomal_loss(model, device):
    def eyes_like(tensor): # eyes means identity matrix
        size = tensor.size()[0]
        return torch.eye(size, out=torch.empty_like(tensor)).to(device)

    def calc_orthonomal_loss(weight):
        mm = torch.mm(weight, torch.transpose(weight, 0, 1))
        return torch.norm(mm - eyes_like(mm))

    orthonomal_loss = torch.tensor(0, dtype=torch.float32).to(device)
    for conv in model.convs[1:]: # in [W^2, W^3, ... , W^L], not include W^1
        n_heads = 1
        if conv.name == 'gcn_conv':
            weights = [conv.conv.lin.weight]
        elif conv.name == 'gat_conv':
            n_heads = conv.conv.heads
            weights = [conv.conv.lin_src.weight]
        # elif conv.name == 'sage_conv':
        #     weights = [conv.conv.lin_r.weight, conv.conv.lin_l.weight]

        for weight in weights:
            orthonomal_loss += (calc_orthonomal_loss(weight) / n_heads / len(weights))

    return orthonomal_loss