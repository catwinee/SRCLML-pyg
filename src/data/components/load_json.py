import json
import torch
from torch import tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import LightGCN

from seqencoder import SequenceEncoder

VERBOSE = True
SHOW_API_NOT_EXIT = False
NODE_FEATURE = 16

def load_node_json(path_dict, encoders=None):
    api_entries, app_entries = [], []
    api_tags_list, app_tags_list = [], []
    api_tags_mapping, app_tags_mapping = {}, {}

    edges_invoke = [[], []]
    edges_app_has_tag = [[], []]
    edges_api_has_tag = [[], []]

    for path in path_dict['api']:
        with open(path, 'r') as f:
            api_entries.extend(json.load(f))
    for path in path_dict['app']:
        with open(path, 'r') as f:
            app_entries.extend(json.load(f))
    if VERBOSE:
        print("json loaded.")

    # 创建标题到索引的映射
    api_mapping = {item['title']: i for i, item in enumerate(api_entries) if item is not None}
    app_mapping = {item['title']: i for i, item in enumerate(app_entries) if item is not None}

    for entry in api_entries:
        # 数据集中，有null对象的存在
        if entry is None:
            continue
        for tag in entry['tags']:
            if tag not in api_tags_list:
                api_index = api_mapping[entry['title']]
                tag_index = len(api_tags_list)

                api_tags_mapping[tag] = tag_index
                api_tags_list.append(tag)

                edges_api_has_tag[0].append(api_index)
                edges_api_has_tag[1].append(tag_index)

    for entry in app_entries:
        if entry is None:
            continue
        for tag in entry['categories']:
            if tag not in app_tags_list:
                app_index = app_mapping[entry['title']]
                tag_index = len(app_tags_list)

                app_tags_mapping[tag] = tag_index
                app_tags_list.append(tag)

                edges_app_has_tag[0].append(app_index)
                edges_app_has_tag[1].append(tag_index)

        for item in entry['related_apis']:
            if item == None:
                continue
            api = item['title']
            if api in api_mapping.keys():
                api_index = api_mapping[api]
                app_index = app_mapping[entry['title']]
                edges_invoke[0].append(app_index)
                edges_invoke[1].append(api_index)
            else:
                # TODO:
                if SHOW_API_NOT_EXIT:
                    print(f"{api} not exit!")

        if encoders is not None:
            # TODO: 添加Transformer来获得description的嵌入
            raise NotImplementedError

    if VERBOSE:
        print("preparation done.")

    data = HeteroData()

    num_apps = len(app_mapping)
    num_apis = len(api_mapping)
    num_app_tags = len(app_tags_list)
    num_api_tags = len(api_tags_list)

    # TODO: 如何确定num_node_dim
    data['app'].x = torch.ones(num_apps, NODE_FEATURE)
    data['api'].x = torch.ones(num_apis, NODE_FEATURE)
    data['app_tags'].x = torch.ones(num_app_tags, NODE_FEATURE)
    data['api_tags'].x = torch.ones(num_api_tags, NODE_FEATURE)

    # TODO: torch.Tensor() 有点慢
    # data['app', 'invoke', 'api'].edge_index = (edges_invoke) # [2, num_edges_invoke]
    # data['app', 'app_has_tag', 'app_tag'].edge_index = (edges_app_has_tag) # [2, num_edges_has_tag]
    # data['api', 'api_has_tag', 'api_tag'].edge_index = (edges_api_has_tag) # [2, num_edges_has_tag]

    data['app', 'invoke', 'api'].edge_index = tensor(edges_invoke, dtype=torch.long) # [2, num_edges_invoke]
    data['app', 'app_has_tag', 'app_tag'].edge_index = tensor(edges_app_has_tag, dtype=torch.long) # [2, num_edges_has_tag]
    data['api', 'api_has_tag', 'api_tag'].edge_index = tensor(edges_api_has_tag, dtype=torch.long) # [2, num_edges_has_tag]
    if VERBOSE:
        print("nodes and edges done.")

    # TODO: 如何使用LightGCN
    LightGCN_invoke = LightGCN(num_apps + num_apis, NODE_FEATURE, 3)
    LightGCN_app_has_tag = LightGCN(num_apps + num_app_tags, NODE_FEATURE, 3)
    LightGCN_api_has_tag = LightGCN(num_apis + num_api_tags, NODE_FEATURE, 3)

    return api_mapping, app_mapping, api_tags_mapping, app_tags_mapping, data

if __name__ == '__main__':
    data_dir = 'src/data/raw/api_mashup'

    path_dict = {
        "api": [data_dir+'/active_apis_data.txt', data_dir+'/deadpool_apis_data.txt'],
        "app": [data_dir+'/active_mashups_data.txt', data_dir+'/deadpool_mashups_data.txt']
    }

    api_mapping, app_mapping, api_tags_mapping, app_tags_mapping, data = load_node_json(path_dict)

    print(len(api_mapping))
    print(data)
