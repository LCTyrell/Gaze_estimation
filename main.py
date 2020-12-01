
import os
import time
import cv2

import logging as log
from openvino.inference_engine import IECore
from argparse import ArgumentParser
from face_detection import Face_detection
from head_pose_estimation import Head_pose
from facial_landmark_detection import Landmark_detection
from gaze_estimation import Gaze_estimation
#from mouse_controller import MouseController

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-mfd", "--model_fd", required=True, type=str,
                        help="Path to the face detection model file without extension (e.g. path/model_name).")
    parser.add_argument("-mhp", "--model_hp", required=True, type=str,
                        help="Path to the head pose model file without extension (e.g. path/model_name).")
    parser.add_argument("-mld", "--model_ld", required=True, type=str,
                        help="Path to the landmark detection model file without extension (e.g. path/model_name).")
    parser.add_argument("-mge", "--model_ge", required=True, type=str,
                        help="Path to the gaze estimation model file without extension (e.g. path/model_name).")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Use 'CAM' for camera or path to video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="Path to cpu extension if needed.")
    parser.add_argument("-df", "--draw_flags", required=False, nargs='+',
                        default=[],
                        help="Flags to draw model(s) output on the video (e.g. fd hp ld ge)"
                             "fd to draw face detection output"
                             "hp to draw head pose output"
                             "ld to draw landmark detection output"
                             "ge to draw gaze estimation output" )
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device: "
                             "CPU, GPU, FPGA, MYRIAD or MULTI (e.g. ""MULTI:CPU(2),GPU(2)"")")
    parser.add_argument("-pt", "--threshold", type=float, default=0.5,
                        help="Minimum inference probability threshold (0.5 by default)")
    return parser


def main():
    """
    Load the network and parse the SSD output.
    """

    args = build_argparser().parse_args()

    model_fd=args.model_fd
    model_hp=args.model_hp
    model_ld=args.model_ld
    model_ge=args.model_ge

    device=args.device
    draw_flags=args.draw_flags
    if args.input=='CAM':
        video_file=0
    else: video_file=args.input

    threshold=args.threshold

    #mc=MouseController(precision='low', speed='fast')

    start_model_load_time=time.time()

    log.info("Creating fd Inference Engine...")
    ie = IECore()

    fd= Face_detection(model_fd, device, threshold)
    hp= Head_pose(model_hp, device, threshold)
    ld= Landmark_detection(model_ld, device, threshold)
    ge=Gaze_estimation(model_ge, device, threshold)

    fd.load_model(ie)
    hp.load_model(ie)
    ld.load_model(ie)
    ge.load_model(ie)

    total_model_load_time = time.time() - start_model_load_time

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fps=15
    out_video = cv2.VideoWriter(os.path.join('results', 'output_video_'+args.device+'.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0
    start_inference_time=time.time()

    try:
        fd.set_initial(initial_w, initial_h)
        hp.set_initial(initial_w, initial_h)
        ge.set_initial(initial_w, initial_h)

        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            coords, image, head_image= fd.predict(frame, draw_flags)

            if coords:
                land_image, left_eye, right_eye, nose= ld.predict(head_image, draw_flags)
                head_pose, pose_image= hp.predict(head_image, nose, draw_flags)
                eye_pose= ge.predict(left_eye, right_eye, head_pose, draw_flags)

            #if counter % 4 ==0:
                #mc.move(-eye_pose[0]/10, eye_pose[1]/10)

            cv2.imshow("Camera_view",cv2.resize(frame,(900,450)))
            #out_video.write(image)

        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join('results', 'stats_ge_'+args.device+'.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__ == '__main__':
    main()
    exit(0)
