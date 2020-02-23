"""
    Python module for converting the trained PyTorch model to ONNX.
    Intel OpenVINO library cannot directly process the state dict
    of PyTorch's trained models. It has to be converted to ONNX prior
    to feeding it in the OpenVINO's Model Optimizer.
"""
# Importing packages
import torch
import torch.nn as nn
import torch.onnx as onnx
import torchvision.models as models
from collections import OrderedDict

# Using CPU
device = torch.device('cpu')

# loading the pretrained densenet 169 model
model = models.densenet169(pretrained=True)

# changing the classifier according to our model, 120 is the total number of classes
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1664, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 120))
]))

model.classifier = classifier

# loading the model's state dictionary
model.load_state_dict(torch.load('fruits.pt', map_location=device))  # converting the model from GPU to CPU

# dummy variable
x = torch.randn(1, 3, 100, 100)         # size of the input image

# exporting to ONNX
onnx.export(model, x, "fruits.onnx", verbose=True, export_params=True)
