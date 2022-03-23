# Import the required libraries
import torch
import torchvision


# MobileNet-v2 with last layer replaced
def get_mobilenet(num_classes):
    mobilenet = torchvision.models.mobilenet_v2()
    mobilenet.classifier[1] = torch.nn.Linear(
        in_features=mobilenet.classifier[1].in_features,
        out_features=num_classes)
    return mobilenet
