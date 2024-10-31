import json
import torch
from torch import tensor
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

def load_from_json(
        encoders=None, 
        emb_dim=32,
        VERBOSE = True,
        SHOW_API_NOT_EXIT = False
    ):
    data_dir = 'src/data/raw/api_mashup'

    path_dict = {
        "api": [data_dir+'/active_apis_data.txt', data_dir+'/deadpool_apis_data.txt'],
        "app": [data_dir+'/active_mashups_data.txt', data_dir+'/deadpool_mashups_data.txt']
    }

    api_entries, app_entries = [], []
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
    # 数据集中存在重复的标题，但是内容不同！
    app_mapping, api_mapping = {}, {}
    app_vis, api_vis = dict(), dict()
    app_cnt, api_cnt = 0, 0
    for item in app_entries:
        if item == None:
            continue
        title = item['title']
        if title not in app_vis.keys():
            app_mapping[title] = app_cnt
            app_cnt += 1
        app_vis[title] = 1
    
    for item in api_entries:
        if item == None:
            continue
        title = item['title']
        if title not in api_vis.keys():
            api_mapping[title] = api_cnt
            api_cnt += 1
        api_vis[title] = 1

    assert len(app_mapping) - 1 == max(app_mapping.values())
    assert len(api_mapping) - 1 == max(api_mapping.values())

    for entry in api_entries:
        # 数据集中，有null对象的存在
        if entry is None:
            continue
        api_index = api_mapping[entry['title']]
        for tag in entry['tags']:
            if tag is None:
                continue
            if tag not in api_tags_mapping.keys():
                tag_index = len(api_tags_mapping)
                api_tags_mapping[tag] = tag_index
            else:
                tag_index = api_tags_mapping[tag]

            edges_api_has_tag[0].append(api_index)
            edges_api_has_tag[1].append(tag_index)

    for entry in app_entries:
        if entry is None:
            continue
        app_index = app_mapping[entry['title']]
        for tag in entry['categories']:
            if tag is None:
                continue
            if tag not in app_tags_mapping:
                tag_index = len(app_tags_mapping)
                app_tags_mapping[tag] = tag_index
            else:
                tag_index = app_tags_mapping[tag]

            edges_app_has_tag[0].append(app_index)
            edges_app_has_tag[1].append(tag_index)

        for item in entry['related_apis']:
            if item == None:
                continue
            api = item['title']
            if api in api_mapping.keys():
                api_index = api_mapping[api]
                edges_invoke[0].append(app_index)
                edges_invoke[1].append(api_index)
            else:
                if SHOW_API_NOT_EXIT:
                    print(f"{api} not exit!")

    if VERBOSE:
        print("preparation done.")

    num_apps = len(app_mapping)
    num_apis = len(api_mapping)
    num_app_tags = len(app_tags_mapping)
    num_api_tags = len(api_tags_mapping)

    app_invoke_api_subgraph = HeteroData()
    app_has_tag_subgraph = HeteroData()
    api_has_tag_subgraph = HeteroData()

    app_invoke_api_subgraph['app'].node_id = torch.arange(num_apps)
    app_invoke_api_subgraph['api'].node_id = torch.arange(num_apis)

    app_has_tag_subgraph['app'].node_id = torch.arange(num_apps)
    app_has_tag_subgraph['app_tag'].node_id = torch.arange(num_app_tags)

    api_has_tag_subgraph['api'].node_id = torch.arange(num_apis)
    api_has_tag_subgraph['api_tag'].node_id = torch.arange(num_api_tags)

    app_invoke_api_subgraph['app', 'invoke', 'api'].edge_index = tensor(edges_invoke, dtype=torch.long) # [2, num_edges_invoke]
    app_has_tag_subgraph['app', 'app_has_tag', 'app_tag'].edge_index = tensor(edges_app_has_tag, dtype=torch.long) # [2, num_edges_has_tag]
    api_has_tag_subgraph['api', 'api_has_tag', 'api_tag'].edge_index = tensor(edges_api_has_tag, dtype=torch.long) # [2, num_edges_has_tag]
    if VERBOSE:
        print("nodes and edges done.")
        print(app_invoke_api_subgraph['app', 'invoke', 'api'].edge_index)

    return app_invoke_api_subgraph, app_has_tag_subgraph, api_has_tag_subgraph

if __name__ == '__main__':
    _1, _2, _3 = load_from_json()
    print(_1, _2, _3)

