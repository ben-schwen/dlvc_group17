import argparse
import cv2
import numpy as np
from cv2 import aruco
from util.aruco_detector import ArucoDetector
from util.dataset import load_dataset

def run_realsense_test(validator: ArucoDetector):
    try:
        for _, frame, depth_map in load_dataset('data/aruco4'):
            found_marker = validator.find_markers(frame)
            if found_marker:
                validator.find_positions()
                validator.find_position_estimates(depth_map, False)
                frame = validator.visualize(frame)
                cv2.imshow('frame', frame)
            else:
                cv2.imshow('frame', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='tool to check realsense dataset accuracy with aruco patterns')

    parser.add_argument('--width', type=int, required=False, default=640,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=360,
                        help='camera capture mode: height')
    parser.add_argument('--path', type=str, required=False, default='data/aruco3',
                        help='camera capture mode: height')
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
    run_realsense_test(aruco_analyzer)

