# model_extraction_interiit

### Training and testing the attacker model with KL divergence loss

Use this command to train and test the attacker model in the vanilla strategy of backpropagating KL divergence loss through attacker model

```python
python run_attacker.py \
  --train_input_dir data/ \
  --train_logits_file logits.pkl \
  --test_input_dir test_data/ \
  --test_logits_file test_logits.pkl \
  --attacker_model_name resnet-lstm \
  --victim_model_name swin-t \
  --epochs 20 \
  --resnet_lstm_trainable_layers 1 \
  --learning_rate 1e-3 \
  --load_test_from_disk
```

where  
`train_input_dir` is the training data directory,  
`test_input_dir` is the testing data directory,  
`train_logits_file` are the logits generated by victim model on training data  
`test_logits_file` are the logits generated by victim model on testing data (Kinetics)  
`attacker_model_name` is swin-t or movinet  
`victim_model_name` is resnet-lstm or c3d, etc.  
`resnet_lstm_trainable_layers` is specific to resnet-lstm and is the number of unfrozen layers of resnet used  
`load_test_from_disk` is to not load more than a batch of images in memory at a time in testing  
`load_train_from_disk` is to not load more than a batch of images in memory at a time in training  