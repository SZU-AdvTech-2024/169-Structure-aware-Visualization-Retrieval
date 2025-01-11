rm -rf cp
rm -rf embedding

mkdir cp
mkdir embedding

CUDA_VISIBLE_DEVICES=0,1 python main_simsiam.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --batch-size 128\
  --fix-pred-lr --workers 24 --epochs 200 --save_epochs 50 --dim 512 --pred-dim 256\
  ../full_VizML+/train_png