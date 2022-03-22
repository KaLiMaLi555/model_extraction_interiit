# Model Extraction: Grey Box Setting

## Requirements

The code is developed and tested using `Python >= 3.6`. To install the requirements:

```bash
pip install -r requirements.txt
```

### To setup the dataset:

Kinetics-400/Kinetics-600

```bash
python Datasets/Kinetics/scripts/kinetics.py [--dir_path SAVE_DIRECTORY]
    [--set_name k400] [--part full] [--max_workers 8] 
    [--extract_frames true] [--extend 16]
```

Once finished, the folder `SAVE_DIRECTORY` should be like this:

``` bash
Kinetics:
    data  
    └── k400
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

UCF-101

```bash 
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
```

## Logit Generation

```bash
python3 get_embeddings.py
```

## Training

```bash
python3 train.py
```

## Evaluation

```bash
python3 test.py
```