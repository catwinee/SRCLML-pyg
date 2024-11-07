from load import load_from_json
from train import train
from test import test

import torch
import torch_geometric
from tqdm import tqdm

from torch import nn
from torch_geometric.nn import LightGCN

from torch.utils.data import DataLoader
from torch.optim import Adam

VERBOSE = False
batch_size = 64

(invoke_subgraph, app_tag_subgraph, api_tag_subgraph,
 ground_truth, description)= load_from_json()

invoke_edge_index = invoke_subgraph['invoke'].edge_index
app_tag_edge_index = app_tag_subgraph['app_has_tag'].edge_index 
api_tag_edge_index = api_tag_subgraph['api_has_tag'].edge_index 

invoke_train_len = int(invoke_edge_index.size(1) * 0.8)
app_tag_train_len = int(app_tag_edge_index.size(1) * 0.8)
api_tag_train_len = int(api_tag_edge_index.size(1) * 0.8)

invoke_train_data = invoke_edge_index[:, :invoke_train_len]
app_tag_train_data = app_tag_edge_index[:, :app_tag_train_len]
api_tag_train_data = api_tag_edge_index[:, :api_tag_train_len]
invoke_test_data = invoke_edge_index[:, invoke_train_len:]
app_tag_test_data = app_tag_edge_index[:, app_tag_train_len:]
api_tag_test_data = api_tag_edge_index[:, api_tag_train_len:]

num_app = invoke_subgraph['app'].num_nodes
num_api = invoke_subgraph['api'].num_nodes
num_app_tag = app_tag_subgraph['app_tag'].num_nodes
num_api_tag = api_tag_subgraph['api_tag'].num_nodes

invoke_train_loader  = DataLoader(
    range(invoke_train_data.size(1)),
    shuffle=True,
    batch_size=batch_size,
)
app_tag_train_loader = DataLoader(
    range(app_tag_train_data.size(1)),
    shuffle=True,
    batch_size=batch_size,
)
api_tag_train_loader = DataLoader(
    range(api_tag_train_data.size(1)),
    shuffle=True,
    batch_size=batch_size,
)

invoke_model = LightGCN(
    num_nodes = num_app + num_api,
    embedding_dim = 32,
    num_layers = 2,
)
invoke_optimizer = Adam(invoke_model.parameters(), lr=0.001)

app_tag_model = LightGCN(
    num_nodes = num_app + num_app_tag,
    embedding_dim = 32,
    num_layers = 2,
)
app_tag_optimizer = Adam(app_tag_model.parameters(), lr=0.001)

api_tag_model = LightGCN(
    num_nodes = num_api + num_api_tag,
    embedding_dim = 32,
    num_layers = 2,
)
api_tag_optimizer = Adam(api_tag_model.parameters(), lr=0.001)


for epoch in tqdm(range(100)):
    loss = train(
        invoke_edge_index,
        invoke_train_loader,
        invoke_model, invoke_optimizer,
        (num_app, num_api)
    ) + train(
        app_tag_edge_index,
        app_tag_train_loader,
        app_tag_model, app_tag_optimizer,
        (num_app, num_app_tag)
    ) + train(
        api_tag_edge_index,
        api_tag_train_loader,
        api_tag_model, api_tag_optimizer,
        (num_api, num_api_tag)
    )
    precision, recall = test(
        edge_index=invoke_edge_index,
        model=invoke_model,
        num_nodes=(num_app, num_api),
        batch_size=batch_size,
        k=20
    )
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Precision@20: '
          f'{precision:.4f}, Recall@20: {recall:.4f}')


