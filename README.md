# Whole-song Generation

[Demo](https://wholesonggen.github.io/) | [Paper](https://openreview.net/forum?id=sn7CYWyavh&noteId=3X6BSBDIPB)

This is the code repository of the paper:

> Ziyu Wang, Lejun Min, and Gus Xia. Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models. ICLR 2024.


# Status
[Apr-03-2024] The current version provides three main usages: 
1. Training all 4 levels of the cascaded models; 
2. Whole-song generation with specified Form (i.e., phrase and key).
3. Whole-song generation with generated Form (i.e., phrase and key).

Currently, generation given prompt (e.g., first several measures) or with external control are not released. These will be properly reformatted in future version.

We only release a portion of the model checkpoints sufficient for testing. The complete set of checkpoints will be released in future versions.

# Downloading data and pre-trained checkpoints
The data and pretrained checkpoints can be downloaded and added to the repository. These are in `data/` (training data), `pretrained_models/` (pretrained VAEs), and `results_default/` (cascaded Diffusion Models). Download them using the corresponding links given in `download_link.txt` files.


# Training
Here are the commands to train four levels of the Diffusion Models. Use `--external` to control whether to use external condition in the training.
```

# form
python train_main.py --mode frm
# optional
python train_main.py --mode frm --multi_label

# counterpoint
python train_main.py --mode ctp --autoreg --mask_bg
# with external control
python train_main.py --mode ctp --autoreg --external --mask_bg

# lead sheet
python train_main.py --mode lsh --autoreg --mask_bg
# with external control
python train_main.py --mode lsh --autoreg --external --mask_bg

# accompaniment
python train_main.py --mode acc --autoreg --mask_bg
# with external control
python train_main.py --mode acc --autoreg --external --mask_bg

```

For a more detailed usage, check
```
python train_main.py -h
```


# Inference

We currently provide functions for whole-song generation with or without form specification. By default, if models are not specified, the models in `results_default` will be used; and the results will be shown in `demo/`.

To generate `n` (e.g., `n=4`) pieces of a given form (e.g., `i4A4A4B8b4A4B8o4` and G major)
```
python inference_whole_song.py --nsample 4 --pstring i4A4A4B8b4A4B8o4 --key 7
```

To generate `n` (e.g., `n=4`) pieces using a generated form:

```
python inference_whole_song.py --nsample 4
```

For a more detailed usage, check

```
python inference_whole_song.py -h
```
