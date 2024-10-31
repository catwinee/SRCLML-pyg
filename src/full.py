from data.components.load_sub_graphs import load_from_json

import torch
import torch_geometric
from tqdm import tqdm

from torch import nn
from torch_geometric.nn import LightGCN
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit

VERBOSE = True
batch_size = 64

invoke_subgraph, app_tag_subgraph, api_tag_subgraph = load_from_json()

invoke_split = RandomLinkSplit(
    num_val = 0,
    num_test = 0.2,
    edge_types = ('app', 'invoke', 'api'),
)
app_has_tag_split = RandomLinkSplit(
    num_val = 0,
    num_test = 0.2,
    edge_types = ('app', 'app_has_tag', 'app_tag'),
)
api_has_tag_split = RandomLinkSplit(
    num_val = 0,
    num_test = 0.2,
    edge_types = ('api', 'api_has_tag', 'api_tag'),
)

if VERBOSE:
    print(app_tag_subgraph)

invoke_train_data, invoke_val_data, invoke_test_data = invoke_split(invoke_subgraph)
app_tag_train_data, app_tag_val_data, app_tag_test_data = app_has_tag_split(app_tag_subgraph)
api_tag_train_data, api_tag_val_data, api_tag_test_data = api_has_tag_split(api_tag_subgraph)

if VERBOSE:
    print("Train data:")
    print("===============================")
    print(invoke_train_data)
    print()

edge_label_index = invoke_train_data["app", "invoke", "api"].edge_label_index
edge_label = invoke_train_data["app", "invoke", "api"].edge_label

invoke_train_loader = LinkNeighborLoader(
    data = invoke_train_data,
    num_neighbors = [30] * 2,
    neg_sampling_ratio = 1,
    edge_label_index=(("app", "invoke", "api"), edge_label_index),
    edge_label=edge_label,
    batch_size=batch_size,
    shuffle=True,
)

if VERBOSE:
    sampled_data = next(iter(invoke_train_loader))
    print(sampled_data)

model = LightGCN(
    num_nodes = invoke_train_data.num_nodes,
    embedding_dim = 32,
    num_layers = 2,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_app, num_api = invoke_subgraph['app'].num_nodes, invoke_subgraph['api'].num_nodes

def train():
    total_loss = total_examples = 0
    num_el = invoke_subgraph['app', 'invoke', 'api'].edge_index.size(1)

    for index in tqdm(invoke_train_loader):
        # Sample positive and negative labels.
        pos_edge_label_index = invoke_subgraph['app', 'invoke', 'api'].edge_index
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_app, num_app + num_api,
                          (num_el, ))
        ], dim=0)
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(invoke_subgraph['app', 'invoke', 'api'].edge_index, edge_label_index).chunk(2)

        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples

for i in range(50):
    print(train())
