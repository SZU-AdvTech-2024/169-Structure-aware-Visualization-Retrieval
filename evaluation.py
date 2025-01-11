#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pickle
import os
import numpy as np
import pandas as pd
import torch as th
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm


parser = argparse.ArgumentParser(description='VizML+ similarity query')
parser.add_argument('-c', '--image_embedding_filename', default="./simsiam/embedding/checkpoint_0200_embedding_test.pkl",
                    help='path to embedding (image)')
parser.add_argument('-e', '--graph_embedding_filename', default="./infograph/embedding/checkpoint_0040_embedding_test.pkl",
                    help='path to embedding (graph/svgs)')
parser.add_argument('-i', '--input_path', default="./full_VizML+/test_graph",
                    help='path to information of graphs (svg)')
parser.add_argument('-p', '--image_folder_path', default="./full_VizML+/test_png",
                    help='path to images')                    

args = parser.parse_args("")


# In[9]:


def read_data(args):

    array_embedding_svg = pickle.load(open(args.graph_embedding_filename, "rb"))

    dict_embedding_ordered_svg = OrderedDict()
    dict_visual_element_count = OrderedDict()

    for i in tqdm(sorted(os.listdir(args.input_path))):
        if (not(".pkl" in i)): continue

        with open(f"{args.input_path}/{i}", "rb") as f:
            dict_temp = pickle.load(f)

        for (k, v) in dict_temp.items():
            if "line" in i:
                dict_embedding_ordered_svg[v.split(".json")[0]] = array_embedding_svg[int(k)]
            elif "scatter" in i:
                dict_embedding_ordered_svg[v.split(".json")[0]] = array_embedding_svg[int(k)]
            else:
                dict_embedding_ordered_svg[v.split(".json")[0]] = array_embedding_svg[int(k)]
        

    for i in tqdm(sorted(os.listdir(args.input_path))):
        if (not(".vec" in i)): continue

        with open(f"{args.input_path}/{i}", "rb") as f:
            dict_temp = pickle.load(f)

        for (k, v) in dict_temp.items():
            dict_visual_element_count[k.split(".json")[0]] = v


    dict_embedding_img = pickle.load(open(args.image_embedding_filename, "rb"))
    dict_embedding_ordered_img = OrderedDict(dict_embedding_img)

    dict_embedding_ordered_all = OrderedDict()


    dict_embedding_ordered_all_class = OrderedDict({
        "bar":[],
        "box":[],
        "line":[],
        "scatter":[]
    })

    list_embedding_svg_aligned = []
    list_embedding_img_aligned = []
    
    prefix = args.image_folder_path.split("_png")[0].split("/")[-1]
    

    
    for (i, (k, v)) in enumerate(dict_embedding_ordered_img.items()):

        new_k = k[0].split(f"{prefix}_png/")[1].split(".png")[0]
        new_k_type = k[0].split(f"{prefix}_png/")[1].split("/")[0] 
        new_k_type = 'bar' if new_k_type == 'histogram' else new_k_type
        
        if new_k not in dict_embedding_ordered_svg:
            print(new_k)
            continue

        dict_embedding_ordered_all_class[new_k_type].append(i)
        dict_embedding_ordered_all[new_k] = [v, dict_embedding_ordered_svg[new_k]]
        list_embedding_svg_aligned.append(dict_embedding_ordered_svg[new_k])
        list_embedding_img_aligned.append(v)

    array_embedding_svg = np.array(list_embedding_svg_aligned)   
    array_embedding_img = np.array(list_embedding_img_aligned)
    
    scaler = MinMaxScaler()
    array_embedding_svg = scaler.fit_transform(array_embedding_svg)
    scaler = MinMaxScaler()
    array_embedding_img = scaler.fit_transform(array_embedding_img)
    
    return array_embedding_svg, array_embedding_img, dict_embedding_ordered_all, dict_visual_element_count


# ## Chart type similarity

# In[3]:


from collections import Counter

def chart_similarity_score(arrays_embedding, dict_embedding_ordered_all, top_k = 5, hist_eq_bar = 1): 
    
    similarity_matrix = np.zeros((arrays_embedding[0].shape[0], arrays_embedding[0].shape[0]), dtype=np.float64)
    
    combine_arrays_embedding = np.concatenate(arrays_embedding, axis=1)
    similarity_matrix += cosine_similarity(combine_arrays_embedding)
    similarity_matrix [np.diag_indices_from(similarity_matrix)] = 0.0

    similarity_matrix /= len(arrays_embedding)
    similar_idx = np.argsort(similarity_matrix)[:,-top_k:]
    if hist_eq_bar:
        list_type = [i.split("/")[0] if i.split("/")[0] != 'histogram' else 'bar'              for i in dict_embedding_ordered_all.keys()]
    else:
        list_type = [i.split("/")[0] for i in dict_embedding_ordered_all.keys()]

    array_result = np.array(list_type)[similar_idx]
    
    list_similarity = np.sum((np.array(list_type)[similar_idx].T == np.array(list_type)).T, axis=1)/top_k

    chart_type_similarity_score = np.mean(list_similarity)
    chart_type_similarity_std = np.std(list_similarity)
    
    return chart_type_similarity_score, chart_type_similarity_std

def visual_element_count_similarity_score(arrays_embedding, dict_visual_element_count, top_k = 5):

    similarity_matrix = np.zeros((arrays_embedding[0].shape[0], arrays_embedding[0].shape[0]), dtype=np.float64)
    
    combine_arrays_embedding = np.concatenate(arrays_embedding, axis=1)
    similarity_matrix += cosine_similarity(combine_arrays_embedding)
    similarity_matrix [np.diag_indices_from(similarity_matrix)] = 0.0
    
    similarity_matrix /= len(arrays_embedding)
    similar_idx = np.argsort(similarity_matrix)[:,-top_k:]
    
    list_vec = []
    for i in list(dict_visual_element_count.values()):
        if i > 0:
            list_vec.append(i)
        else:
            list_vec.append(i+1)
        
    array_vec_similar = np.array(list_vec)[similar_idx]
    
    array_vec = np.array([list_vec]).T
      
    list_vec_mean = np.mean(np.abs((array_vec_similar - array_vec)/array_vec), axis=1)
    
    return np.mean(list_vec_mean), np.std(list_vec_mean)


# In[11]:


list_result = []


array_embedding_svg, array_embedding_img, dict_embedding_ordered_all, dict_visual_element_count  = read_data(args)
for top_k in [1, 5, 10, 20]:
    chart_type_similarity_score, chart_type_similarity_std = chart_similarity_score([array_embedding_img, array_embedding_svg], dict_embedding_ordered_all, hist_eq_bar = 1, top_k = top_k)
    vec_score, vec_std = visual_element_count_similarity_score([array_embedding_img, array_embedding_svg], dict_visual_element_count, top_k = top_k)
    list_result.append({
        "top_k": top_k,
        "type_sim_score": chart_type_similarity_score,
        "type_sim_std": chart_type_similarity_std,
        "vec_score": vec_score,
        "vec_std": vec_std
    })


# In[17]:


print(pd.DataFrame(list_result).groupby(['top_k']).mean())
