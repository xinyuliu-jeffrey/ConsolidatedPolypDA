# Feature-level Adaptation of ConsolidatedPolypDA

## Install

Please check [INSTALL.md](INSTALL.md) for instructions

## Data Preparation

```
  mkdir -p datasets/polyps
  ln -s /path_to_source_dataset/annotations datasets/polyps/annotations
  ln -s /path_to_source_dataset/images datasets/polyps/images
  ln -s /path_to_target_dataset/annotations datasets/polyps/annotations
  ln -s /path_to_target_dataset/images datasets/polyps/images
```
Or you may directly modify the path_catalog.py at ./maskrcnn_benchmark/config/path_catalog.py

## Training and Testing

Coming Soon

## Pretrained Models and Logs

Coming Soon

## Acknowledgements
This repo is built on top of [DA Faster R-CNN](https://github.com/krumo/Domain-Adaptive-Faster-RCNN-PyTorch) and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)