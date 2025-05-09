from .gnn import GCN, GAT
    

def load_net(cfg, **kwargs):

    if cfg.base_gnn == 'GCN':
        return GCN(cfg)
    elif cfg.base_gnn == 'GAT':
        return GAT(cfg)