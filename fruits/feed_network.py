import os
import argparse

### Load the necessary libraries
from openvino.inference_engine import IECore, IENetwork

CPU_EXTENSION = "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension.dylib"

def get_args():
    """
    Gets the arguments from the command line
    """
    parser = argparse.ArgumentParser("Load the IR model into the Inference Engine")
    #---- Create the descriptions of the commands
    m_desc = "The location of the model XML file"

    #---- Create the arguments
    parser.add_argument("-m", help=m_desc)
    args = parser.parse_args()

    return args

def load_to_IE(model_xml):
    ### Load the Inference Engine API
    model_bin = os.path.splitext(model_xml)[0] + ".bin"    # might have used just model_xml.split()[0]
    plugin = IECore()
    
    ### Load IR files into their related class
    net = IENetwork(model=model_xml, weights=model_bin)
    
    ### Add a cpu extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    plugin.add_extension(extension_path=CPU_EXTENSION, device_name="CPU")
    
    ### Get the supported layers of the network
    supported_layers = plugin.query_network(network=net, device_name="CPU")
    print("supported layers: ", len(supported_layers))
    print("total layers:", len(net.layers.keys()))

    unsupported_layers = len(net.layers) - len(supported_layers)
    if unsupported_layers != 0:
        print("Unsupported layers found: {}".format(unsupported_layers))
        exit(1)
    
    ### Load the network into the inference engine
    exec_net = plugin.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.")

    return

def main():
    args = get_args()
    load_to_IE(args.m)

if __name__ == "__main__":
    main()
