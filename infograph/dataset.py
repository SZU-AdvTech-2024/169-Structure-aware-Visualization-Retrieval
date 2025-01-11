import dgl
import os
import torch as th
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs

from tqdm import tqdm
from collections import Counter

dict_vis_type = {
    "bar": 0,
    "box": 1,
    "heatmap": 2,
    "histogram": 3,
    "line": 4,
    "scatter": 5
}


class VizMLPlusDataset(DGLDataset):
    def __init__(self, input_path):
        self.input_path = input_path
        self.graphs = []
        self.labels = []

        super().__init__(name='VizML+')

    def process(self):
        for i in tqdm(sorted(os.listdir(self.input_path))):
            if (not(".bin" in i)): continue
            
            temp_graphs, _ = load_graphs(f"{self.input_path}/{i}")
            label = dict_vis_type[i.split(".")[0]]

            for j in temp_graphs:
                if j.ndata['vector'].shape[1] < 30:
                    j.ndata['vector'] = th.cat([j.ndata['vector'], th.zeros((j.ndata['vector'].shape[0], 5))], 1).to(th.float32)
                else:
                    j.ndata['vector'] = j.ndata['vector'].to(th.float32)
            
            self.graphs += temp_graphs
            self.labels += [label for _ in range(len(temp_graphs))]

        print(f"Graphs: {len(self.graphs)}; Labels: {Counter(self.labels)}")

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)