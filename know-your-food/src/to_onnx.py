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

# Using CPU
device = torch.device('cpu')

# Loading the pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Changing the classifier according to our model, 120 is the total number of classes
classifier = nn.Sequential(
    (nn.Linear(2048, 1000)),
    (nn.ReLU()),
    (nn.Linear(1000, 120))
)

model.fc = classifier

# Loading the model's state dictionary
model.load_state_dict(torch.load('fruits.pt', map_location=device))  # converting the model from GPU to CPU

# Set the model to inference mode
model.eval()

# dummy variable
x = torch.randn(1, 3, 224, 224)         # size of the input image

# exporting to ONNX
onnx.export(model, x, "fruits.onnx", verbose=True, export_params=True)
