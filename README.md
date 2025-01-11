### How to run the code
1. run `./svg_processing.sh` and `python graph_construction.py` in `svg_processing` folder
2. run `./train.sh` in `simsiam` and `infograph` folders
3. run `./embed.sh` in `simsiam` and `infograph` folders after training models
4. run `python evaluation.py` under the main directory

### Required packages
- Python: DGL 0.6.1, PyTorch 1.9.0

- JavaScript: d3-regression 1.3.9, svg-parser 2.0.4

### Reference
[1] SimSiam: https://github.com/facebookresearch/simsiam

[2] InfoGraph-DGL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/infograph