Colab Notebook: [Link](https://colab.research.google.com/drive/1uRYFLCy-JLisbgxjRzRRXw_MBr5F7zZz?usp=sharing)

Clone the repo

```
git clone -b conv3d https://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@github.com/manideep1108/model_extraction_interiit
```

Install requirements.txt

```
pip install -r model_extraction_interiit/conv3d/requirements.txt
```

Run generate_data.py
```
python model_extraction_interiit/conv3d/train.py --input_dir = <path> --logits_file = <path> --epochs = <value> --batch_size = <value>
```