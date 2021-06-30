import numpy as np
import cv2
from util.estimators import DepthEstimator
import pandas as pd


def draw_text(frame, title):
    frame = frame.copy()
    tl = 2
    tf = 1
    t_size = cv2.getTextSize(
        title, 0, fontScale=tl / 3, thickness=tf)[0]
    width = t_size[0]
    height = t_size[1]

    c1 = tuple(np.array([5, 5]).astype(int))
    c2 = tuple(np.array([width+15, height+20]).astype(int))

    frame = cv2.rectangle(
        frame, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)  # filled
    frame = cv2.putText(frame, title, (c1[0]+5, c1[1]+height+5), 0, tl / 3,
                        [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)
    return frame

depth_estimator = DepthEstimator()
root = 'data/office'
file_names = open(root+'/names.txt', 'r')
names = file_names.readlines()

count = 0
rmse_logs = np.zeros((len(names), 4))
for i, name in enumerate(names):
    name = name[:-5]
    rgb = cv2.imread(f'{root}/rgb/{name}.png')
    rgb = draw_text(rgb, 'RGB-Image')
    depth = cv2.imread(f'{root}/depth/{name}.png', -1)*0.0010000000474974513

    adabins = np.load(f'{root}/adabins/{name}.npy')
    adabins_rmse = depth_estimator.compute_errors(depth, adabins)
    adabins = depth_estimator.colorize_pred(adabins)
    adabins = draw_text(adabins, 'adabins')

    midas_small = np.load(f'{root}/midas-small/{name}.npy')
    midas_small_rmse = depth_estimator.compute_errors(depth, midas_small, True)
    midas_small = depth_estimator.colorize_pred(midas_small, inverse=True)
    midas_small = draw_text(midas_small, 'MiDaS small')

    midas_hybrid = np.load(f'{root}/midas-hybrid/{name}.npy')
    midas_hybrid_rmse = depth_estimator.compute_errors(
        depth, midas_hybrid, True)
    midas_hybrid = depth_estimator.colorize_pred(midas_hybrid, inverse=True)
    midas_hybrid = draw_text(midas_hybrid, 'DPT Hybrid')

    midas_large = np.load(f'{root}/midas-large/{name}.npy')
    midas_large_rmse = depth_estimator.compute_errors(depth, midas_large, True)
    midas_large = depth_estimator.colorize_pred(midas_large, inverse=True)
    midas_large = draw_text(midas_large, 'DPT Large')

    depth = depth_estimator.colorize_gt(depth)
    depth = draw_text(depth, 'Intel RealSense D455')
    frame1 = cv2.hconcat([rgb, depth, adabins])
    frame2 = cv2.hconcat([midas_small, midas_hybrid, midas_large])
    frame = cv2.vconcat([frame1, frame2])

    print(f'adabins: {adabins_rmse:.3f} MiDas small: {midas_small_rmse:.3f} DPT Hybrid: {midas_hybrid_rmse:.3f} DPT Large: {midas_large_rmse:.3f} ')
    rmse_logs[i, 0] = adabins_rmse
    rmse_logs[i, 1] = midas_small_rmse
    rmse_logs[i, 2] = midas_hybrid_rmse
    rmse_logs[i, 3] = midas_large_rmse

    cv2.imwrite(f'results/video/img{count:05d}.png', frame)
    count += 1

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pd.DataFrame(rmse_logs,columns=['adabins', 'small', 'hybrid', 'large']).to_csv('result.csv', index=False)
