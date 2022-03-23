# Bosch's Model Extraction Attack For Video Classification

## Requirements

The code is developed and tested using `Python >= 3.6`. To install the requirements:

```bash
pip install -r requirements.txt
```

<hr/>

## Grey Box Setting

### Dataset Setup

* #### Kinetics-400/Kinetics-600

```bash
python Datasets/Kinetics/scripts/kinetics.py             \
    [--dir_path <save_dir_path>]                         \
    [--set_name k400/k600]                               \
    [--part full/train/test/val/replacement/annotations] \
    [--max_workers 8]                                    \
    [--extract_frames true]                              \
    [--extend 16]
```

Once finished, the folder `save_dir_path` should be like this:

``` bash
Kinetics:
    data  
    └── k400
        └── train
            └── video_1
                └── frame_1
                └── frame_2
                ...
            └── video_2
                └── frame_1
                └── frame_2
                ...
            ...
            └── val
            └── test
            └── annotations
            └── replacement
```

* #### UCF-101

``` bash 
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
``` 

### Logit Generation

```bash
python3 get_embeddings.py
```

### Training

```bash
python3 train.py
```

### Evaluation

```bash
python3 test.py
```

<hr/>

## Black Box Setting

Download the weights of swin-t. URL: [Link](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth)

model_name = `swint` or `movinet`

### Data Free Model Extraction

* Modify the `config/params_dfme_<model_name>.yaml` file to use the options as desired for Swin-T or MoViNet respectively.

* Train using command `python DFME/train.py --config path-to-config-yaml`

### Data Free Model Extraction with Conditional GAN

* #### Pretraining cGAN

  * Modify the `config/params_pretrain_<model_name>.yaml` file to use the options as desired for Swin-T or MoViNet respectively.

  * Run experiment using `python cGAN/pretrain_generator.py --config path-to-config-yaml`  

* #### Training Threat Model with Single Frame

  * Modify the `config/params_cgan_<model_name>.yaml` file to use the options as desired for Swin-T or MoViNet respectively.

  * Run experiment using `python cGAN/train.py --config path-to-config-yaml`
