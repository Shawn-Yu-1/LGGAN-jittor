## Contents

  - [Introduce](#Introduce)
  - [Dataset Preparation](#Dataset-Preparation)
  - [Trainset](#trainset)
  
  
## Introduce

this is the LGGAN of jittor edition, which for a competition of Jittor AI

## Dataset Preparation
you can use the custom.py to load your own dataset for training

## Trainset
1. Prepare dataset.
2. Change several parameters and then run `train_[dataset].sh` for training.
There are many options you can specify. To specify the number of GPUs to utilize, use `--gpu_ids`. If you want to use the second and third GPUs for example, use `--gpu_ids 1,2`.
3. Testing is similar to testing pretrained models. Use `--results_dir` to specify the output directory. `--how_many` will specify the maximum number of images to generate. By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

