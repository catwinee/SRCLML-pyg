from typing import Dict, List, Set

import torch
from torch import nn
import json

from torch_geometric.data import HeteroData
from seqencoder import SequenceEncoder

# src/data/components/load_json.py
from load_json import load_node_json

data_dir = 'src/data/raw/api_mashup'

path_dict = {
    "api": [data_dir+'/active_apis_data.txt', data_dir+'/deadpool_apis_data.txt'],
    "app": [data_dir+'/active_mashups_data.txt', data_dir+'/deadpool_mashups_data.txt']
}

api_mapping, app_mapping, api_tags_mapping, app_tags_mapping, data = load_node_json(path_dict)
