import argparse
import cv2
import numpy as np

from handle_model_output import handle_output, preprocessing
from inference import Network

MODEL_OUTPUT = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 
                'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 
                'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 
                'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 
                'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 
                'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 
                'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 
                'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 
                'Chestnut', 'Clementine', 'Cocos', 'Dates', 'Eggplant', 'Ginger Root', 
                'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 
                'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 
                'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 
                'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 
                'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 
                'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 
                'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 
                'Peach Flat', 'Pear', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 
                'Pear Monster', 'Pear Red', 'Pear Williams', 'Pepino', 'Pepper Green', 
                'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 
                'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 
                'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 
                'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 
                'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 
                'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Maroon', 
                'Tomato Yellow', 'Walnut']

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Know your Food - Arguments")

    # -- Create the descriptions for the commands
    c_desc = "CPU extension file location, if applicable"
    d_desc = "Device, if not CPU (GPU, FPGA, MYRIAD)"
    i_desc = "The location of the input file"
    m_desc = "The location of the model XML file"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    optional.add_argument("-m", help=m_desc, default="fruits.xml")
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    args = parser.parse_args()

    return args


def create_output_image(image, output):
    '''
    creates an output image showing the result of inference.
    '''    
    return image

def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(args.m, args.d, args.c)
    net_input_shape = plugin.get_input_shape()

    # print(net_input_shape)
    # Read the input image    
    image = cv2.imread(args.i)
    # print('net_input_shape is ' + str(net_input_shape))

    # Preprocess the input image
    preprocessed_image = preprocessing(image, net_input_shape[2], net_input_shape[3])
    # print(preprocessed_image.shape)
    # Perform synchronous inference on the image
    plugin.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = plugin.extract_output()
    # print(output)
    processed_output = handle_output(output)
    print("processed output", processed_output)

    print(MODEL_OUTPUT[processed_output])
    
    # Create an output image based on network
    output_image = create_output_image(image, processed_output)

    # Save down the resulting image
    cv2.imwrite("outputs/output.png", output_image)


def main():
    args = get_args()
    perform_inference(args)


if __name__ == "__main__":
    main()
