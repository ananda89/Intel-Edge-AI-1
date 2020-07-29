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
        default=0.6,
        help="Probability threshold for detections filtering"
        "(0.6 by default)",
    )
    parser.add_argument(
        "-v",
        "--visualize",
        required=False,
        nargs="+",
        default=[],
        help="Argument to specify if any type of visulaization of bounding box,"
        "display of other stats are required. Multiple arguments (correspnding"
        "to diiferent model) can be chained. Possible arguments:"
        "fd: face detection bbox,"
        "fld: bboxs on both images,"
        "hpe: head pose (displays three angles),"
        "ge: shows the gaze vectors",
    )

    return parser


def infer_on_stream(args):
    input_file = args.input_path
    display_items = args.visualize

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

    # keep track of the frames
    frame_number = 0
    for flag, frame in inputfeeder.next_batch():
        if not flag:
            break
        # keep track of frames passed
        frame_number += 1
        key_pressed = cv2.waitKey(60)

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
        left_eye_image, right_eye_image, eye_coordinates = fld_model.predict(
            face_image
        )
        print("Left and right eye images")
        print(left_eye_image.shape)
        print(right_eye_image.shape)

        # get the coordinates for mouse controller
        *mouse_coordinates, gaze_vector = ge_model.predict(
            left_eye_image, right_eye_image, head_pose_angles
        )
        print("Coordinates of the mouse pointer")
        print(mouse_coordinates)

        # check if display stats are requested, if so, show them
        display_frame = frame.copy()
        if len(display_items) != 0:
            if "fd" in display_items:
                cv2.rectangle(
                    display_frame,
                    (face_coordinates[0], face_coordinates[1]),
                    (face_coordinates[2], face_coordinates[3]),
                    (32, 32, 32),
                    2,
                )
            if "fld" in display_items:
                # showing bbox on left eye
                cv2.rectangle(
                    display_frame,
                    (eye_coordinates[0][0], eye_coordinates[0][1]),
                    (eye_coordinates[0][2], eye_coordinates[0][3]),
                    (220, 20, 60),
                    2,
                )

                # showing bbox on right eye
                cv2.rectangle(
                    display_frame,
                    (eye_coordinates[1][0], eye_coordinates[1][1]),
                    (eye_coordinates[1][2], eye_coordinates[1][3]),
                    (220, 20, 60),
                    2,
                )

            if "hpe" in display_items:
                # show yaw, pitch and roll angles on the frame
                text = f"""yaw:{head_pose_angles[0]:.1f}, 
                pitch:{head_pose_angles[1]:.1f}, 
                roll:{head_pose_angles[2]:.1f}"""
                cv2.putText(
                    display_frame,
                    text,
                    (5, 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(255, 255, 255),
                    thickness=1,
                )

            # if 'ge' in display_items:
            #     # show the gaze vector
            #     left
            #     cv2.line

        if frame_number % 5 == 0:
            cv2.imshow("video", cv2.resize(display_frame, (500, 500)))

        if frame_number % 5 == 0:
            mc.move(mouse_coordinates[0], mouse_coordinates[1])

        if key_pressed == 27:
            print("Exit key is pressed...exiting!")
            break

    # closing the video stream
    inputfeeder.close()

    print(f"Total frames: {frame_number}")


def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)


if __name__ == "__main__":
    main()
