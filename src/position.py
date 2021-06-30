from adabins.infer import InferenceHelper
import argparse
import cv2
import numpy as np
from cv2 import aruco
from util.aruco_detector import ArucoDetector
from PIL import Image

inferHelper = InferenceHelper()

def get_depth(frame):
    frame_rgb = frame[:, :, ::-1]
    im_pil = Image.fromarray(frame_rgb)
    centers, pred = inferHelper.predict_pil(im_pil)
    return pred[0][0]

def run_estimation(source, width, height, validator: ArucoDetector, make_video=True):
    cap = cv2.VideoCapture(source)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            more, frame = cap.read()
            found_marker = validator.find_markers(frame)
            if found_marker:
                validator.find_positions()
                depth_map = get_depth(frame)
                validator.find_position_estimates(depth_map, True)
                frame = validator.visualize(frame)
                cv2.imshow('frame', frame)
            else:
                cv2.imshow('frame', frame)
            if not more:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CPOP tool to visually validate a camera calibration')

    parser.add_argument('--width', type=int, required=False, default=640,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=360,
                        help='camera capture mode: height')
    parser.add_argument('--source', required=False, default=0,
                        help='camera source, can be a video')
    parser.add_argument('--camera-model', type=str, required=False, default='default',
                        help='camera model name (for storage)')
    parser.add_argument('--marker-size', type=float, required=False, default=0.16,
                        help='length of the aruco marker in meters (required for camera position calculation)')

    args = parser.parse_args()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    mtx = np.load(
        f'data/{args.camera_model}_camera_matrix_{args.width}x{args.height}.npy')
    dist = np.load(
        f'data/{args.camera_model}_dist_coeffs_{args.width}x{args.height}.npy')

    print(f'camera_matrix: {mtx}')
    print(f'dist_coeffs: {dist}')

    aruco_analyzer = ArucoDetector(mtx, dist, aruco_dict, args.marker_size)
    run_estimation(args.source, args.width, args.height, aruco_analyzer)