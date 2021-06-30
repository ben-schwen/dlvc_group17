import matplotlib
from PIL import Image
from adabins.infer import InferenceHelper
import argparse
import cv2
import numpy as np
from cv2 import aruco
from util.dataset import load_dataset
import os
import torch

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


class DepthEstimator():

    def __init__(self, min_depth=1e-3, max_depth=10):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def get_depth(self):
        return NotImplemented

    def colorize(self, gt, pred, cmap='magma_r'):
        return self.colorize_gt(gt, cmap), self.colorize_pred(pred, cmap)

    def colorize_gt(self, gt, cmap='magma_r'):
        return self._colorize_frame(gt, cmap)

    def colorize_pred(self, pred, cmap='magma_r', inverse=False):
        return self._colorize_frame(pred, cmap, inverse)

    def _colorize_frame(self, value, cmap, inverse=False):
        invalid_mask = value == -1

        mask = value != 0
        if not np.any(mask):
            return value[:,:,:3]
        # if inverse:
        #     value[mask] = 1/value[mask]

        vmax = value[mask].max()
        vmin = value[mask].min()

        # normalize
        if vmin != vmax:
            value = (value - vmin) / (vmax - vmin)  # vmin..vmax
        else:
            # Avoid 0-division
            value = value * 0.

        cmapper = matplotlib.cm.get_cmap(cmap)
        value = cmapper(value, bytes=True)  # (nxmx4)
        value[invalid_mask] = 255
        img = value[:, :, :3]

        return img

    def compute_errors(self, gt, pred, inverse=False):
        valid_mask = np.logical_and(gt > self.min_depth, gt < self.max_depth)
        valid_mask = np.logical_and(valid_mask, pred > self.min_depth)

        # eval_mask = np.zeros(valid_mask.shape)
        # eval_mask[40:320, 40:600] = 1
        # valid_mask = np.logical_and(valid_mask, eval_mask)

        gt = gt[valid_mask]
        pred = pred[valid_mask]
        if inverse:
            gt = 1/gt

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        return rmse_log


class AdabinsDepthEstimator(DepthEstimator):
    def __init__(self):
        super().__init__()
        self.inferHelper = InferenceHelper()

    def get_depth(self, frame):
        frame_rgb = frame[:, :, ::-1]
        im_pil = Image.fromarray(frame_rgb)
        centers, pred = self.inferHelper.predict_pil(im_pil)
        return pred[0][0]


class MidasDepthEstimator(DepthEstimator):
    def __init__(self, model_type):
        super().__init__()
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def get_depth(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        return output


def get_midas(model_type):
    return MidasDepthEstimator(model_type)


def get_adabins():
    return AdabinsDepthEstimator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='tool to estimate and store depth')

    parser.add_argument('--width', type=int, required=False, default=640,
                        help='camera capture mode: width')
    parser.add_argument('--height', type=int, required=False, default=360,
                        help='camera capture mode: height')
    parser.add_argument('--path', required=False, default='office',
                        help='path of the intel realsense depth camera data')
    parser.add_argument('--camera-model', type=str, required=False, default='default',
                        help='camera model name (for storage)')
    parser.add_argument('--estimator', type=str, required=False, default='adabins',
                        help='depth estimation network (adabins)')
    args = parser.parse_args()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    mtx = np.load(
        f'data/{args.camera_model}_camera_matrix_{args.width}x{args.height}.npy')
    dist = np.load(
        f'data/{args.camera_model}_dist_coeffs_{args.width}x{args.height}.npy')

    print(f'camera_matrix: {mtx}')
    print(f'dist_coeffs: {dist}')

    if args.estimator == 'adabins':
        depth_estimator = get_adabins()
    elif args.estimator == 'midas-large':
        depth_estimator = get_midas('DPT_Large')
    elif args.estimator == 'midas-hybrid':
        depth_estimator = get_midas('DPT_Hybrid')
    elif args.estimator == 'midas-small':
        depth_estimator = get_midas('MiDaS_small')
    else:
        raise Exception('Unknown depth estimator')

    estimation_path = args.path+f'/{args.estimator}'
    if not os.path.exists(estimation_path):
        os.makedirs(estimation_path)

    loader = load_dataset(args.path)
    for name, frame, depth_map in loader:
        depth_pred = depth_estimator.get_depth(frame)
        # cv2.imwrite(f"{estimation_path}/{name}", depth_pred)
        np.save(f"{estimation_path}/{name}.npy", depth_pred)
        #errors = depth_estimator.compute_errors(depth_map, depth_pred)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # run_estimation(loader, depth_estimator)
