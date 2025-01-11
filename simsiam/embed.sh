for i in 200
do
  python main_embedding.py \
    -a resnet50 \
    --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 --b 32\
    --pretrained "./cp/checkpoint_0${i}.pth.tar" --evaluate --dataset train\
    ../full_VizML+/train_png

  python main_embedding.py \
    -a resnet50 \
    --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 --b 32\
    --pretrained "./cp/checkpoint_0${i}.pth.tar" --evaluate --dataset test\
    ../full_VizML+/test_png
done

