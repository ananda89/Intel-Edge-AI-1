import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Defining some transforms to process the input image suitable for the 
# input to inference engine
transform = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))
                                                      ])

def handle_output(output):
    '''
    Handles the output of the asl recognition model.
    Returns one integer = prob: the argmax of softmax output.
    '''
    return np.argmax(output.flatten()) 

def preprocessing(img):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    img = transform(img)
    img = torch.unsqueeze(img, 0)       # reshaping the image by adding 1 (batch_size) in dim=0
    img = img.numpy()
    return img