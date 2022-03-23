# Running Blackbox Model Extraction

1. Install all requirements ```pip install -q -r requirements.txt```
2. Download the weights of swin-t. URL: https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth

## Data Free Model Extraction

Modify the `config/params_dfme_swint.yaml` or `config/params_dfme_movinet.yaml` file to use the options as desired for Swin-T or MoViNet respectively and then ```python DFME/train.py --config path-to-config-yaml```  
OR  
Create a new config file to use the options as desired and then ```python DFME/train.py --config path-to-config-yaml```  
NOTE: Make sure to properly set the model checkpoints path in the config.

## Data Free Model Extraction with Conditional GAN

### Pretraining cGAN
Modify the `config/params_pretrain_swint.yaml` or `config/params_pretrain_movinet.yaml` file to use the options as desired for Swin-T or MoViNet respectively and then ```python cGAN/pretrain_generator.py --config path-to-config-yaml```  
OR  
Create a new config file to use the options as desired and then ```python cGAN/pretrain_generator.py --config path-to-config-yaml```  
NOTE: Make sure to properly set the path of datasets and model checkpoint path in the config.

### Training Threat Model with Single Frame

Modify the `config/params_cgan_swint.yaml` or `config/params_cgan_movinet.yaml` file to use the options as desired for Swin-T or MoViNet respectively and then ```python cGAN/train.py --config path-to-config-yaml```  
OR  
Create a new config file to use the options as desired and then ```python cGAN/train.py --config path-to-config-yaml```  
NOTE: Make sure to properly set the path of datasets and model checkpoint path in the config.
