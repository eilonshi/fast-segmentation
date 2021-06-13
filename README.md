# Tevel Urban Semantic Segmentation 

Our usage of [BiSeNetV2](https://arxiv.org/abs/1808.00897) for semantic segmentation in an urban images case

## platform
My platform is like this: 
* ubuntu 20.04
* nvidia GTX 1080 Ti
* cuda 11.2
* miniconda python 3.8
* pytorch 1.7.1
* torchvision 0.8.2

## get start
With a pretrained weight, you can run inference on a single image like this: 
```
$ python src/models/inference.py --model bisenetv2 --weight-path models/path/to/your/weights.pth --img-path ./example.png
```
This would run inference on the image and save the result image to `data/inference_results`

## prepare dataset

If you want to train on your own dataset, you should generate annotation files first with the format like this: 
```
munster_000002_000019_leftImg8bit.png,munster_000002_000019_gtFine_labelIds.png
frankfurt_000001_079206_leftImg8bit.png,frankfurt_000001_079206_gtFine_labelIds.png
...
```
Each line is a pair of training sample and ground truth image path, which are separated by a single comma `,`.   
Then you need to change the field of `im_root` and `train/val_im_anns` in the configuration files.

## train
In order to train the model, you can run command like this: 
```
$ export CUDA_VISIBLE_DEVICES=0,1

# if you want to train with apex
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --model bisenetv2 # or bisenetv1

# if you want to train with pytorch fp16 feature from torch 1.6
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train_amp.py --model bisenetv2 # or bisenetv1
```

Note that though `bisenetv2` has fewer flops, it requires much more training iterations. The the training time of `bisenetv1` is shorter.

## finetune from trained model
You can also load the trained model weights and finetune from it:
```
$ export CUDA_VISIBLE_DEVICES=0,1
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --finetune-from ./res/model_final.pth --model bisenetv2 # or bisenetv1

# same with pytorch fp16 feature
$ python -m torch.distributed.launch --nproc_per_node=2 tools/train_amp.py --finetune-from ./res/model_final.pth --model bisenetv2 # or bisenetv1
```

## eval pretrained models
You can also evaluate a trained model like this: 
```
$ python tools/evaluate.py --model bisenetv1 --weight-path /path/to/your/weight.pth
```
