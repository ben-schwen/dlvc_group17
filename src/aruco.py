import argparse
from util.aruco_detector import ArucoDetector
import numpy as np
from cv2 import aruco
import cv2

def run_aruco_detection(source, width, height, aruco_analyzer: ArucoDetector):
    cap = cv2.VideoCapture(source)
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        while True:
            more, frame = cap.read()
            if not more:
                break
            found_markers = aruco_analyzer.find_markers(frame)
            if found_markers:
                aruco_analyzer.find_positions()
                frame = aruco_analyzer.visualize(frame)
                cv2.imshow('frame', frame)
            else:
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='tool to find aruco patterns')

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

    run_aruco_detection(args.source, args.width, args.height, aruco_analyzer)
