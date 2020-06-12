"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


# MQTT server environment variables
import os
import sys
import time
import socket
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# # TOPICS information
# PERSON_TOPIC = "person"
# DURATION_TOPIC = "person/duration"


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser(description="Run inference an on input video",
                            allow_abbrev=True)
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def draw_bbox(threshold, output, frame, width, height):
    people_count = 0
    coord1 = None
    coord2 = None
    for box in output[0][0]:  # Output shape is 1x1x100x7
        conf = box[2]
        if (box[1] == 1) and (conf > threshold):
            prob = conf * 100
            color = (200, 10, 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            prob = 'human: {:.2f}%'.format(prob)
            xmin = int(box[3]*width)
            ymin = int(box[4]*height)
            xmax = int(box[5]*width)
            ymax = int(box[6]*height)
            coord1 = (xmin, ymin)
            coord2 = (xmax, ymax)
            cv2.rectangle(frame, coord1, coord2, color, 2)
            cv2.putText(frame, prob, (xmin+2, ymin+12), font,
                        fontScale=0.4, thickness=1, color=(36, 67, 223))
            people_count += 1
    return frame, people_count


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # declaring variables to count the people and duration
    total_number = 0
    last_number = 0
    missed_number = 0
    start = 0
    duration = 0
    frame_number = 0

    # Initialise the class
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    # This applies only for faster rcnn since it outputs two things for
    # input shape: image: [1, 3] and image tensor: [1, 3, 600, 600]
    # We need image tensor
    # input_shape = net_input_shape['image_tensor']

    ### TODO: Handle the input stream ###
    single_image_mode = False

    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
    else:
        # Check the input value
        assert os.path.isfile(args.input), "Input file doesn't exist..."

    captured = cv2.VideoCapture(args.input)
    captured.open(args.input)

    # Grab the shape of the input
    width = int(captured.get(3))
    height = int(captured.get(4))

    # Processing the video
    # Create a video writer for the output video
    # if not single_image_mode:
    #     # out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width, height))    # for linux
    #     out = cv2.VideoWriter('out_frcnn.mp4', cv2.VideoWriter_fourcc(
    #         'M', 'J', 'P', 'G'), 30, (width, height))   # for Mac
    # else:
    #     out = None

    ### TODO: Loop until stream is over ###
    while captured.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = captured.read()
        frame_number += 1
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        p_frame = cv2.resize(
            frame, (net_input_shape[3], net_input_shape[2]))  # for SSD model
        # p_frame = cv2.resize(
        #     frame, (input_shape[3], input_shape[2]))  # for faster rcnn
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Input to the network (only required for faster rcnn)
        # network_input_data = {'image_tensor': p_frame,
        #                       'image_info': p_frame.shape[1:]}

        # request id for making inferences
        request_id = 0

        ### TODO: Start asynchronous inference for specified request ###
        # Start asynchronous inference for specified request.
        infer_start = time.time()
        infer_network.exec_net(request_id, p_frame)     # for SSD
        # infer_network.exec_net(
        #     request_id, network_input_data)    # for faster rcnn

        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:
            ### TODO: Get the results of the inference request ###
            det_time = time.time() - infer_start

            result = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###

            # Draw bounding box
            frame, current_number = draw_bbox(
                prob_threshold, result, frame, width, height)
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

            # When a person enters the frame
            if current_number > last_number:
                start = time.time()
                total_number += current_number - last_number
                client.publish("person", json.dumps({"total": total_number}))

            # when a person leaves the frame
            if current_number == 0 and last_number != 0:
                missed_number += 1

                # wait for few frames to make sure the person has actually left the frame
                # this number should be bigger for SSD because of high false negatives
                # missing frame threshold
                if missed_number >= 30:   # use 30 for SSD and 5 for faster rcnn
                    duration = int(time.time() - start)
                    client.publish("person/duration",
                                   json.dumps({"duration": duration}))
                    # resetting the dropped frames
                    missed_number = 0
                    # updating the last number
                    last_number = current_number
            else:
                # publishing the results
                client.publish("person", json.dumps({"count": current_number}))
                # updating the last number
                last_number = current_number

            # Write out the frame
            # out.write(frame)
            # Break if escape key pressed
            if key_pressed == 27:
                break
        ### TODO: Send the frame to the FFMPEG server ###
        # Resize the frame according to the video
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite("output_image.jpg", p_frame)

    # Release the capture and destroy any OpenCV windows
    captured.release()
    cv2.destroyAllWindows()
    # TODO: Disconnect from MQTT
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
