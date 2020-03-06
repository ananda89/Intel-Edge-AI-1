import argparse
import numpy as np
import cv2
from PIL import Image       # Using PIL instead of OpenCV
from handle_model_output import handle_output, preprocessing
from get_nutritional_data import get_nutritional_data
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
    k_desc = "API Key value"
    id_desc = "API Id value"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    optional.add_argument("-m", help=m_desc, default="../models/fruits.xml")
    optional.add_argument("-c", help=c_desc, default=None)
    optional.add_argument("-d", help=d_desc, default="CPU")
    optional.add_argument("-id", help=id_desc, default=None)
    optional.add_argument("-k", help=k_desc, default=None)
    
    args = parser.parse_args()

    return args


def create_output_image(nutritional_info):
    '''
    creates an output image showing the result of inference.
    '''    
    img = cv2.imread('../images/output-background.jpg')
    
    y0, dy = 100, 50
    for i, line in enumerate(nutritional_info.split('\n')):
    	# Only picking the top 20 nutrients
        if i < 20: 
            y = y0 + i*dy          
            cv2.putText(img, line, (900, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img

def perform_inference(args):
    '''
    Performs inference on an input image, given a model.
    '''
    # Create a Network for using the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(args.m, args.d, args.c)
    net_input_shape = plugin.get_input_shape()
    
    # Read the input image
    image = Image.open(args.i)
    
    # Preprocess the input image
    preprocessed_image = preprocessing(image)

    # Perform synchronous inference on the image
    plugin.sync_inference(preprocessed_image)

    # Obtain the output of the inference request
    output = plugin.extract_output()

    processed_output = handle_output(output)    
    processed_output = MODEL_OUTPUT[processed_output].split()[0]
    ret_string = get_nutritional_data(processed_output, args.id, args.k)

    # Create an output image based on network
    output_image = create_output_image(ret_string)

    # Save down the resulting image
    cv2.imwrite("../outputs/output.png", output_image)

def main():
    args = get_args()
    perform_inference(args)


if __name__ == "__main__":
    main()
