"""
    Python script to test the performance of the pretrained pytorch model.
    It loads the pretrained model from the state dictionary containing the 
    trained model weights. It then takes the input as an RGB image, resize 
    it to 100x100, applies some transforms to it to make it suitable for 
    the input to the model and then ouput the predicted class label.
"""
# Importing packages
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import json

# Using CPU
device = torch.device('cpu')

# loading the pretrained densenet 169 model
model = models.resnet50(pretrained=True)

# changing the classifier according to our model, 120 is the total number of classes
classifier = nn.Sequential(
    (nn.Linear(2048, 1000)),
    (nn.ReLU()),
    (nn.Linear(1000, 120))
)

model.fc = classifier

# loading the model's state dictionary
model.load_state_dict(torch.load('fruits.pt', map_location=device))  # converting the model from GPU to CPU

# get the classes and write tham in a file
# data_dir = 'project-intel-edge-ai/fruits/fruits-360_dataset/fruits-360/'
data_dir = 'fruits-360_dataset/fruits-360/'

train_dir = os.path.join(data_dir, 'Training')
test_dir = os.path.join(data_dir, 'Test')

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    classes = classes[1:]
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    return classes, idx_to_class


class_names, idx_to_class = find_classes(test_dir)

# output the dictionary for the class names
with open("idx_to_class.json", "w") as f:
    json.dump(idx_to_class, f)

# Modifying the images according to ImageNet images
# Defining Image loader
loader = transforms.Compose([transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), 
                            (0.5, 0.5, 0.5))])

# Defining the size for custom images
height = 100
width = 100

# defining the path of the image
img_path = 'sweet_potato.jpg'
img = Image.open(img_path)

img = img.resize((height, width))

img = loader(img)
img = torch.unsqueeze(img, 0)
print(img.shape)

# Setting the model to evaluation mode for inference
model.eval()
result = model(img)

print(result.shape)         # should be a (1 x num_classes) array
_, preds_tensor = torch.max(result, 1)

# Getting the predictionsfrom the model output.
preds = np.squeeze(preds_tensor.numpy()) 

# Printing out the class corresponding to the prediction
print(class_names[preds])