from __future__ import division
import torch
from torch import nn
import resnext


def generate_model(pretrain_path=None, n_finetune_classes=None):

    from resnext import get_fine_tuning_parameters
    model = resnext.resnet101(
        num_classes=101,
        shortcut_type='B',
        cardinality=32,
        sample_size=224,
        sample_duration=16,
        input_channels=3,
        output_layers=[])

    model = model.cuda()
    model = nn.DataParallel(model)

    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path)

        model.load_state_dict(pretrain['state_dict'])
        model.module.fc = nn.Linear(model.module.fc.in_features, n_finetune_classes)
        model.module.fc = model.module.fc.cuda()

        ft_begin_index=4
        parameters = get_fine_tuning_parameters(model, ft_begin_index)
        return model, parameters

    return model, model.parameters()