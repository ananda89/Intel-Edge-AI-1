"""
Actual python script to run the application.
"""
import sys
import os
import time
import cv2
import numpy as np

from argparse import ArgumentParser

# model scripts
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksModel
from head_pose_estimation import HeadPoseEstimationModel
from gaze_estimation import GazeEstimationModel
from mouse_controller import MouseController
from input_feeder import InputFeeder


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser(
        description="Run the application on demo video", allow_abbrev=True
    )
    # arguments for the paths to various models
    parser.add_argument(
        "-fd",
        "--face_detection",
        required=True,
        type=str,
        help="Path to the xml file of the trained face-detection model.",
    )
    parser.add_argument(
        "-fld",
        "--facial_landmarks",
        required=True,
        type=str,
        help="Path to the xml file of the trained facial landmarks detection model.",
    )
    parser.add_argument(
        "-hpe",
        "--head_pose",
        required=True,
        type=str,
        help="Path to the xml file of the trained head pose estimation model",
    )
    parser.add_argument(
        "-ge",
        "--gaze",
        required=True,
        type=str,
        help="Path to the xml file of the trained gaze estimation model",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        type=str,
        help="Path to input image or video file or webcam (CAM)",
    )
    parser.add_argument(
        "-l",
        "--cpu_extension",
        required=False,
        type=str,
        default=None,
        help="MKLDNN (CPU)-targeted custom layers."
        "Absolute path to a shared library with the"
        "kernels impl.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="CPU",
        help="Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (CPU by default)",
    )
    parser.add_argument(
        "-pt",
        "--prob_threshold",
        type=float,
        default=0.5,
        help="Probability threshold for detections filtering"
        "(0.5 by default)",
    )

    return parser


def infer_on_stream(args):
    input_file = args.input_path

    if input_file == "CAM":
        inputfeeder = InputFeeder("cam")
    elif input_file.endswith(".jpg") or input_file.endswith("bmp"):
        inputfeeder = InputFeeder("image", input_file)
    elif input_file.endswith(".mp4"):
        inputfeeder = InputFeeder("video", input_file)
    else:
        assert os.path.isfile(input_file), "Input file doesn't exist..."
        sys.exit(1)

    # storing all the model paths in a dictionary
    model_paths = {
        "face_detection": args.face_detection,
        "facial_landmarks_detection": args.facial_landmarks,
        "head_pose_estimation": args.head_pose,
        "gaze_estimation": args.gaze,
    }
    # checking if all the model file paths are valid
    for model_name in model_paths.keys():
        if not os.path.isfile(model_paths[model_name] + ".xml"):
            print(
                f"Path to the xml file for the model: {model_name} doesn't exist..."
            )
            sys.exit(1)

    # load data from input feeder
    inputfeeder.load_data()

    # instantiating mouse controller
    mc = MouseController(precision="medium", speed="fast")

    # instantiating each model
    fd_model = FaceDetectionModel(
        model_paths["face_detection"],
        device=args.device,
        threshold=args.prob_threshold,
        extensions=args.cpu_extension,
    )
    fld_model = FacialLandmarksModel(
        model_paths["facial_landmarks_detection"],
        device=args.device,
        extensions=args.cpu_extension,
    )
    hpe_model = HeadPoseEstimationModel(
        model_paths["head_pose_estimation"],
        device=args.device,
        extensions=args.cpu_extension,
    )
    ge_model = GazeEstimationModel(
        model_paths["gaze_estimation"],
        device=args.device,
        extensions=args.cpu_extension,
    )

    # load the models and check for unsupported layers
    for model_obj in (fd_model, fld_model, hpe_model, ge_model):
        model_obj.load_model()
        model_obj.check_model()

    frame_number = 0
    for flag, frame in inputfeeder.next_batch():
        if not flag:
            break
        # keep track of frames passed
        frame_number += 1
        key_pressed = cv2.waitKey(60)

        if frame_number % 5 == 0:
            cv2.imshow("video", cv2.resize(frame, (500, 500)))
        # detect the face in the frame
        face_coordinates, face_image = fd_model.predict(frame.copy())
        if face_coordinates == 0:
            print("No face is detected")
            continue
        print(face_image.shape)
        print("Coordinates of the person's face in the frame")
        print(face_coordinates)

        # detect the head pose in the face image
        head_pose_angles = hpe_model.predict(face_image)

        print("Person's head pose angles in the frame")
        print(head_pose_angles)

        # get the left and right eye images
        left_eye_image, right_eye_image = fld_model.predict(face_image)
        print("Left and right eye images")
        print(left_eye_image.shape)
        print(right_eye_image.shape)

        # get the coordinates for mouse controller
        mouse_coordinates = ge_model.predict(
            left_eye_image, right_eye_image, head_pose_angles
        )
        print("Coordinates of the mouse pointer")
        print(mouse_coordinates)
        if frame_number % 5 == 0:
            mc.move(mouse_coordinates[0], mouse_coordinates[1])

        if key_pressed == 27:
            print("Exit key is pressed...exiting!")
            break
        inputfeeder.close()


def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)


if __name__ == "__main__":
    main()
