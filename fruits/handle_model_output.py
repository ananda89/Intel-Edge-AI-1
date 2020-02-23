import cv2
import numpy as np

def handle_output(output):
    '''
    Handles the output of the asl recognition model.
    Returns one integer = prob: the argmax of softmax output.
    '''
    # model_pred = np.argmax(output.flatten())  
    # print(model_pred)
    print(output)
    return np.argmax(output.flatten()) 


def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1,3, height, width)
        
    return image