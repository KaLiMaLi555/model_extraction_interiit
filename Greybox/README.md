# Model Extraction: Grey Box Setting

## Requirements

The code is developed and tested using `Python >= 3.6`. To install the requirements:

```bash
pip install -r requirements.txt
```

### To setup the dataset:
VOC:
```bash
bash data/setup_voc.bash /path-to-data-directory

Once finished, the folder `data` should be like this:
VOC:
```
    data   
    └── VOCdevkit
        └── VOC2012
            ├── JPEGImages
            ├── SegmentationClassAug
            ├── Annotations
            ├── ImageSets
            ├── BgMaskfromBoxes
            └── Generation
                ├── Y_crf
                └── Y_ret
```

## Training

The training procedure is divided into 3 stages and example commands for each have been given below. Hyperparameters can be adjusted accordingly in the corresponding configuration files.

```bash
python3 stage1.py --config-file configs/stage1.yml --gpu-id 0
```

## Evaluation

To evaluate the model on the validation set of Pascal VOC 2012 dataset before and after Dense CRF processing change the `DATA.MODE` parameter to `val` in the corresponding config file:

```bash
python3 stage3.py --config-file configs/stage3_vgg.yml --gpu-id 0
```

## Quantitative Results

We achieve the following results:

- Comparison of pseudo labels on the PASCAL VOC 2012 validation set in terms of mIoU

| **Method**          | **Original Author's Results** | **Our Results** |
|:-------------------:|:-----------------------------:|:---------------:|
| **GAP**             | 76.1                          | 75.5            |
| **BAP Ycrf w/o u0** | 77.8                          | 77              |
| **BAP Ycrf**        | 79.2                          | 78.8            |
| **BAP Yret**        | 69.9                          | 69.9            |
| **BAP Ycrf & Yret** | 68.2                          | 72.7            |