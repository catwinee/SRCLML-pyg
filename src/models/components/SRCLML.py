from typing import Any, Dict, Tuple

import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import LightGCN
from torch_geometric.loader import LinkNeighborLoader
from torch.nn import Transformer

class SRCLML(nn.Module):
    def __init__(
            self, 
            invoke_subgraph, app_tag_subgraph, api_tag_subgraph,
            embd_dim, num_layers
        ) -> None:
        super(SRCLML, self).__init__()

        # 有待优化
        self.invoke_subgraph = invoke_subgraph
        self.app_tag_subgraph = app_tag_subgraph
        self.api_tag_subgraph = api_tag_subgraph

        self.num_app = invoke_subgraph['app'].num_nodes
        self.num_api = invoke_subgraph['api'].num_nodes
        self.num_app_tag = app_tag_subgraph['app_tag'].num_nodes
        self.num_api_tag = api_tag_subgraph['api_tag'].num_nodes

        self.app_embed = torch.randn(self.num_app, embd_dim)
        self.api_embed = torch.randn(self.num_api, embd_dim)
        self.app_tag_embed = torch.randn(self.num_app_tag, embd_dim)
        self.api_tag_embed = torch.randn(self.num_api_tag, embd_dim)

        # TODO: 需要调整
        self.app_api_gcn = LightGCN(self.num_app + self.num_api, embd_dim, num_layers)
        self.app_tag_gcn = LightGCN(self.num_app + self.num_app_tag, embd_dim, num_layers)
        self.api_tag_gcn = LightGCN(self.num_app + self.num_api_tag, embd_dim, num_layers)

        # self.transformer = Transformer()

    def forward(self, 
            app_api_data, app_tag_data, api_tag_data,
        ) -> torch.Tensor:
        # TODO: forward 中loss计算需要根据full.py中的实现进行修改
        self.app_embed = self.app_api_gcn(self.app_embed, app_api_data)
        self.api_embed = self.app_api_gcn(self.api_embed, app_api_data)
        self.app_tag_embed = self.app_tag_gcn(self.app_tag_embed, app_tag_data)
        self.api_tag_embed = self.api_tag_gcn(self.api_tag_embed, api_tag_data)

        return self.loss_demo(self.invoke_subgraph, ('app', 'invoke', 'api'), app_api_data, self.app_api_gcn, self.app_num, self.api_num)

    def loss_func(self, **kwargs):
        return self.loss_demo(**kwargs)

    def loss_demo(self, subgraph, edge_type, loader, model, num_first, num_second):
        total_loss = total_examples = 0
        num_el = subgraph[edge_type].edge_index.size(1)

        for index in tqdm(loader):
            # Sample positive and negative labels.
            pos_edge_label_index = subgraph[edge_type]['app', 'invoke', 'api'].edge_index
            neg_edge_label_index = torch.stack([
                pos_edge_label_index[0],
                torch.randint(num_first, num_first + num_second,
                              (num_el, ))
            ], dim=0)
            edge_label_index = torch.cat([
                pos_edge_label_index,
                neg_edge_label_index,
            ], dim=1)

            # optimizer要用吗？
            # optimizer.zero_grad()
            pos_rank, neg_rank = model(subgraph[edge_type].edge_index, edge_label_index).chunk(2)

            loss = model.recommendation_loss(
                pos_rank,
                neg_rank,
                node_id=edge_label_index.unique(),
            )
            loss.backward()
            # optimizer.step()

            total_loss += float(loss) * pos_rank.numel()
            total_examples += pos_rank.numel()

        return total_loss / total_examples


if __name__ == "__main__":
    _ = SRCLML()
