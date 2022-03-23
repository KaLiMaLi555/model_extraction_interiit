## How to run

1. Install all requirements ```pip install -q -r requirements.txt```
2. Download the weights of swin-t or movinet as required

### Data Free Model Extraction with Videos

Modify the `config/params_dfme_swint.yaml` or `config/params_dfme_movinet.yaml` file to use the options as desired for swin-t or movinet respectively and then ```python DFME/train.py```  
OR  
Create a new config file to use the options as desired and then ```python DFME/train.py --config path-to-config-yaml```  
NOTE: Make sure to properly set the path of datasets and model checkpoint path in the config.

### Conditional GAN pretraining

Modify the `config/params_pretrain_swint.yaml` or `config/params_pretrain_movinet.yaml` file to use the options as desired for swin-t or movinet respectively and then ```python cgan/.py```  
OR  
Create a new config file to use the options as desired and then ```python cgan/train.py --config path-to-config-yaml```  
NOTE: Make sure to properly set the path of datasets and model checkpoint path in the config.

### Conditional GAN training on single frame

Modify the `config/params_cgan_swint.yaml` or `config/params_cgan_movinet.yaml` file to use the options as desired for swin-t or movinet respectively and then ```python cgan/pretrain_generator.py```  
OR  
Create a new config file to use the options as desired and then ```python cgan/pretrain_generator.py --config path-to-config-yaml```  
NOTE: Make sure to properly set the path of datasets and model checkpoint path in the config.
