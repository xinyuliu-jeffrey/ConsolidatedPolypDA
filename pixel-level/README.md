# Pixel-level Adaptation of ConsolidatedPolypDA

## How to use GFDA to transfer your own source dataset to target dataset

Make sure your images are all in .png format. Other formats are not supported yet but is easy to implement by modifing the code.

```
  git clone https://github.com/xinyuliu-jeffrey/ConsolidatedPolypDA.git
  cd ConsolidatedPolypDA
  cd pixel-level
  python image_similarity.py --src_path ./path_to_source_image --trg_path ./path_to_target_image
  python style_transfer.py --src_path ./path_to_source_image --trg_path ./path_to_target_image --save_path ./empty_path_for_saving_images
```

## Acknowledgements
This repo is built on top of [FDA](https://github.com/YanchaoYang/FDA)