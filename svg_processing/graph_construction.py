#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import json
# import argparse
import matplotlib.pyplot as plt

from networkx.readwrite import json_graph
import dgl
import numpy as np

import os
from tqdm import tqdm
import torch as th


# In[2]:


direct_feature_key = ['stroke_red', 'stroke_green', 'stroke_blue', 'fill_red', 'fill_green', 'fill_blue',
                      'fill_opacity', 'stroke_opacity', 'stroke_width', 'cx', 'cy', 'area', 'dx', 'dy', 'length',
                      'num_vertex', 'delta_x', 'delta_y', 'feature_trend_0', 'feature_trend_1', 'feature_trend_2',
                      'feature_trend_3', 'feature_trend_4']

type_embedding = {
    "text": [1, 0, 0, 0, 0, 0, 0],
    "path": [0, 1, 0, 0, 0, 0, 0],
    "g": [0, 0, 1, 0, 0, 0, 0],
    "svg": [0, 0, 0, 1, 0, 0, 0],
    "rect": [0, 0, 0, 0, 1, 0, 0],
    "other": [0, 0, 0, 0, 0, 1, 0],
    "point": [0, 0, 0, 0, 0, 0, 1]
}

def convert_feature_to_vector(dict_feature):
    feature_vector = []
    for i in direct_feature_key:
        if i in dict_feature:
            feature_vector.append(dict_feature[i])
        else:
            feature_vector.append(0)
    
    if dict_feature['type'] in type_embedding.keys():
        feature_vector += type_embedding[dict_feature['type']]
    else:
        feature_vector += type_embedding['other']
    
    return feature_vector


# In[ ]:


input_path = "../full_VizML+/test_json"
output_path = "../full_VizML+/test_graph"
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info

dict_error = {}
count = 0

for i in sorted(os.listdir(input_path)):

    list_graph = []
    dict_error[i] = 0
    dict_id_filename = {}
    
    dict_visual_element_count = {}
    
    if not os.path.isdir(f"{input_path}/{i}"):
        continue
    
    for j in tqdm(sorted(os.listdir(f"{input_path}/{i}"))):
        try:
            with open(f"{input_path}/{i}/{j}", 'r') as f:
                data = json.load(f)

            G_0 = json_graph.tree_graph(data)
            count_remove = 0
            list_nodes = G_0.nodes

            # remove <g> node with only one child
            for n in list_nodes:
                if G_0.out_degree(n) == 1 and 'tagName' not in G_0.nodes[n]:
                    continue
                if G_0.out_degree(n) == 1 and G_0.nodes[n]['tagName'] == 'g':
                    count_remove += 1
                    parent = next(G_0.predecessors(n))
                    G_0 = nx.contracted_edge(G_0, (parent, n))
    
            G = nx.Graph(G_0)
            
            vec = list(dict(G.degree).values())[1:].count(1)

            # add self-loop
            G.add_edges_from([[i, i] for i in G.nodes])

            # add link between leaf nodes
            for n in G.nodes:
                if 'features' not in G.nodes[n]:
                    continue
                if 'pre_node' in G_0.nodes[n]['features']:
                    pre_node = G_0.nodes[n]['features']['pre_node']
                    G.add_edges_from([[n, pre_node]])

            # add feature vector
            for k in G.nodes:
                if "features" not in G.nodes[k]:
                    feature_vector = [0 for _ in range(len(direct_feature_key))]
                    if 'tagName' in G.nodes[k].keys() and G.nodes[k]['tagName'] in type_embedding.keys():
                        feature_vector += type_embedding[G.nodes[k]['tagName']]
                    else:
                        if G.nodes[k]['type'] != 'root': print(G.nodes[k])
                        feature_vector += type_embedding['other']
                    G.nodes[k]['vector'] = th.nan_to_num(th.tensor(feature_vector), nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    G.nodes[k]['vector'] = th.nan_to_num(th.tensor(convert_feature_to_vector(G.nodes[k]['features'])), nan=0.0, posinf=0.0, neginf=0.0)

                G.nodes[k]['graph_o_id'] = count
                G.nodes[k]['node_id'] = int(k.split("_")[-1])
            
            dict_id_filename[str(count)] = f"{i}/{j}"
            
            # transform to dgl graph
            list_graph.append(dgl.from_networkx(G, node_attrs=['vector', 'graph_o_id', 'node_id']))
            graph_id = count
            
            # save vec
            dict_visual_element_count[dict_id_filename[str(graph_id)]] = vec
            
            count += 1
        except:
            print(f"{i}/{j}")
            dict_error[i] += 1
        
    # save list_graphs
    save_graphs(f"{output_path}/{i}.bin", list_graph)
    save_info(f"{output_path}/{i}.pkl", dict_id_filename)
    # save vec count
    save_info(f"{output_path}/{i}.vec", dict_visual_element_count)

# %%
