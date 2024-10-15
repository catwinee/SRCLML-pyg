from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import dgl
from dgl import DGLGraph

from torch_geometric.nn import LightGCN
from torch.nn import Transformer


class SRCLML(nn.Module):
    def __init__(
            self, 
            num_app, num_serv, num_t_app, num_t_serv,
            embd_dim, num_layers,
        ) -> None:
        super(SRCLML, self).__init__()

        self.app_serv_gcn = LightGCN(num_app + num_serv, embd_dim, num_layers)
        self.app_tag_gcn = LightGCN(num_app + num_t_app, embd_dim, num_layers)
        self.serv_tag_gcn = LightGCN(num_app + num_t_serv, embd_dim, num_layers)

        self.transformer = Transformer()

    def forward(self, 
            app_serv_g: DGLGraph, app_tag_g: DGLGraph, serv_tag_g: DGLGraph, 
            text_feature: torch.Tensor
        ):

        self.app_serv_gcn(app_serv_g)
        self.app_tag_gcn(app_tag_g)
        self.serv_tag_gcn(serv_tag_g)

        self.transformer(text_feature)

    def loss_func(self):
        pass

if __name__ == "__main__":
    pass
