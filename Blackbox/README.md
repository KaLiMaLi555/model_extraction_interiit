## How to run

1. Install all requirements ```pip install -q -r requirements.txt```
2. Download the weights of swin-t or movinet
3. Modify the config/params.yaml file to use the options as desired and then ```python train_threat.py```
OR
Create a new config file to use the options as desired and then ```python train_threat.py --config path-to-config-yaml```
Properly set the path of datasets and model checkpoint path in the config.
