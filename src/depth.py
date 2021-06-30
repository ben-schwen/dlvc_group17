import argparse
import cv2
import numpy as np
from cv2 import aruco
from util.dataset import load_dataset, load_estimation_dataset
from util.estimators import DepthEstimator, get_adabins, get_midas
import os

# def run_estimation(loader, depth_estimator: DepthEstimator, use_saved, save_results):

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='tool to view and save depth estimations')

    parser.add_argument('--width', type=int, required=False, default=640,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=360,
                        help='camera capture mode: height')
    parser.add_argument('--path', required=False, default='office',
                        help='path of the intel realsense depth camera data')
    parser.add_argument('--camera-model', type=str, required=False, default='default',
                        help='camera model name (for storage)')
    parser.add_argument('--estimator', type=str, required=False, default='adabins',
                        help='depth estimation network (adabins, manequin)')
    parser.add_argument('--use-saved', required=False, action='store_true',
                        help='use stored depth estimation if specified')
    parser.add_argument('--save-results', required=False, action='store_true',
                        help='use stored depth estimation if specified')
    args = parser.parse_args()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    mtx = np.load(
        f'data/{args.camera_model}_camera_matrix_{args.width}x{args.height}.npy')
    dist = np.load(
        f'data/{args.camera_model}_dist_coeffs_{args.width}x{args.height}.npy')

    print(f'camera_matrix: {mtx}')
    print(f'dist_coeffs: {dist}')

    loader = load_dataset(args.path)
    if args.use_saved:
        print(f'use stored values of {args.estimator}')
        depth_estimator = DepthEstimator()
        loader = load_estimation_dataset(args.path, args.estimator)
    elif args.estimator == 'adabins':
        print(f'estimate depth with of {args.estimator}')
        depth_estimator = get_adabins()
    elif args.estimator == 'midas-large':
        depth_estimator = get_midas('DPT_Large')
    elif args.estimator == 'midas-hybrid':
        depth_estimator = get_midas('DPT_Hybrid')
    elif args.estimator == 'midas-small':
        depth_estimator = get_midas('MiDaS_small')
    else:
        raise Exception('Unknown depth estimator')
    
    if args.use_saved:
        if args.save_results:
            visualization_path = args.path+f'/vis'
            if not os.path.exists(visualization_path):
                os.makedirs(visualization_path)
        for file_name, frame, depth_map, depth_estimation in loader:
            errors = depth_estimator.compute_errors(
                depth_map, depth_estimation)
            depth_map, depth_estimation = depth_estimator.colorize(
                depth_map, depth_estimation)
            cv2.imshow('frame', frame)
            cv2.imshow('depth prediction', depth_estimation)
            cv2.imshow('depth map', depth_map)
            cv2.imwrite(f'{visualization_path}/{file_name}.png', depth_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        if args.save_results:
            estimation_path = args.path+f'/{args.estimator}'
            if not os.path.exists(estimation_path):
                os.makedirs(estimation_path)
        for name, frame, depth_map in loader:
            depth_estimation = depth_estimator.get_depth(frame)
            errors = depth_estimator.compute_errors(
                depth_map, depth_estimation)
            if args.save_results:
                np.save(f"{estimation_path}/{name}.npy", depth_estimation)
            depth_map, depth_estimation = depth_estimator.colorize(
                depth_map, depth_estimation)
            cv2.imshow('frame', frame)
            cv2.imshow('depth prediction', depth_estimation)
            cv2.imshow('depth map', depth_map)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
