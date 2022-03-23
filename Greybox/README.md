# Model Extraction: Grey Box Setting

## Requirements

The code is developed and tested using `Python >= 3.6`. To install the requirements:

```bash
pip install -r requirements.txt
```

## Dataset Setup

### Kinetics-400/Kinetics-600


```bash
find . -name '*.sh' -type f -exec chmod +x {} + 
python Datasets/Kinetics/scripts/kinetics.py [--dir_path <SAVE_DIRECTORY_PATH>]
    [--set_name <k400/k600>] [--part <full/train/test/val/replacement/annotations>] [--max_workers <value>] 
    [--extract_frames <true/false>] [--extend <value>]
```
- Update permissions of all scripts in the `Datasets/Kinetics/scripts` directory.
- Downloads compressed dataset into `./k400_targz/k600_targz` folder
- Extracts the compressed videos into `./k400/k600` folder
- Extracts frames from the videos at 1 fps using ffmpeg
- Duplicates frames uniformly to extend videos to the desired length

Once finished, the folder `SAVE_DIRECTORY` should be like this:

``` bash
Kinetics:
    data  
    └── k400/k600
        └── train
            ├── video_1
                └── frame_1
                └── frame_2
                ...
            ├── video_2
                └── frame_1
                └── frame_2
                ...
            ...
            └── val
            └── test
            └── annotations
            └── replacement
```

### UCF-101

```bash 
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x ./UCF101.rar
``` 

## Logit Generation

```bash
python3 get_embeddings.py [--<param_1> <value> .. <param_n> <value>]
```
All the parameters under `embedding` in `/config/params.json` can be passed as command line arguments to overwrite config during run time.

## Training

```bash
python3 train.py [--<param_1> <value> .. <param_n> <value>]
```
All the parameters under `experiment` in `/config/params.json` can be passed as command line arguments to overwrite config during run time.

## Evaluation

```bash
python3 test.py [--<param_1> <value> .. <param_n> <value>]
```

All the parameters under `test` in `/config/params.json` can be passed as command line arguments to overwrite config during run time.
