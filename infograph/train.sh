rm -rf cp
rm -rf embedding

mkdir cp
mkdir embedding

python unsupervised.py --epochs 40 --lr 0.001 --hid_dim 256 --n_layers 2 --save_interval 10 --gpu 1 --input_path ../full_VizML+/train_graph