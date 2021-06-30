import cv2
import numpy as np
from cv2 import aruco
from statistics import mean

class PointEstimates:
    def __init__(self, image_points, camera_points):
        self.image_points = image_points
        self.camera_points = camera_points

class ArucoDetector():
    def __init__(self, mtx, dist, aruco_dict, marker_size):
        self.mtx = mtx
        self.dist = dist
        self.aruco_dict = aruco_dict
        self.marker_size = marker_size
        self.position_estimates = []

    def find_markers(self, frame):
        found_markers = False
        mtx = self.mtx
        dist = self.dist
        aruco_dict = self.aruco_dict
        marker_size = self.marker_size
        # Grayscale the image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            # Find aruco markers in the query image
            corners, ids, rejected = aruco.detectMarkers(
                gray, dictionary=aruco_dict)
            if corners:
                self.corners = corners
                self.ids = ids
                self.rvecs, self.tvecs, self.obj_points = aruco.estimatePoseSingleMarkers(
                    corners=corners,
                    markerLength=marker_size,
                    cameraMatrix=mtx,
                    distCoeffs=dist,
                    rvecs=None,
                    tvecs=None
                )
                found_markers = True
        except cv2.error as e:
            print(e)
        return found_markers

    def find_positions(self):
        image_points = []
        positions = []
        for id, corner, rvec, tvec in zip(self.ids, self.corners, self.rvecs, self.tvecs):
            corner = corner[0]
            image_point, _ = cv2.projectPoints(
                np.float32([[0, 0, 0]]), rvec, tvec, self.mtx, self.dist)
            image_point = image_point[0][0]
            image_points.append(image_point.astype(int))
            position = tvec[0]
            positions.append(position)
        self.image_points = image_points
        self.positions = positions
        return True

    def find_position_estimates(self, depth_map, calibrate=False):
        self.position_estimates = []
        scale = 1
        if calibrate:
            scale_factors = []
            for id, image_point, position in zip(self.ids, self.image_points, self.positions):
                if id[0] == 0:
                    depth = depth_map[image_point[1], image_point[0]]
                    scale = np.linalg.norm(position)/depth
                    scale_factors.append(scale)
            if scale_factors:
                scale = mean(scale_factors)
            else:
                return False
        for image_point, position in zip(self.image_points, self.positions):
            postition_estimate = self.get_xyz(image_point.astype(
                float), depth_map[image_point[1], image_point[0]])*scale
            self.position_estimates.append(postition_estimate)
        return True

    def get_xyz(self, image_point, depth):
        # calculate the 3d direction of the ray in camera coordinate frame
        mtx, dist = self.mtx, self.dist
        image_point_norm = cv2.undistortPoints(
            image_point, mtx, dist)[0][0]
        ray_dir_cam = np.array(
            [image_point_norm[0], image_point_norm[1], 1], dtype=float)
        # TODO check if correct:
        ray_dir_cam = ray_dir_cam/np.linalg.norm(ray_dir_cam)
        return ray_dir_cam*depth

    def visualize(self, frame):
        frame = aruco.drawDetectedMarkers(frame, self.corners, self.ids)
        frame = self.vis_cubes(frame)
        frame = self.vis_positions(frame)
        return frame

    def vis_positions(self, frame):
        for i in range(len(self.image_points)):
            position = self.positions[i]
            position_strings = [
                f' marker: {self.ids[i][0]}',
                f' aruco: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})']
            if self.position_estimates:
                pos_estimate = self.position_estimates[i]
                position_strings.append(
                    f' depth: ({pos_estimate[0]:.2f}, {pos_estimate[1]:.2f}, {pos_estimate[2]:.2f})')
            frame = self.draw_text(
                frame, self.image_points[i], position_strings)
        return frame

    def vis_cubes(self, frame) -> np.array:
        marker_size = self.marker_size
        mtx = self.mtx
        dist = self.dist
        rvecs = self.rvecs
        tvecs = self.tvecs

        half_size = marker_size / 2
        axis = np.float32([
            [-half_size, -half_size, 0],
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, marker_size],
            [-half_size, half_size, marker_size],
            [half_size, half_size, marker_size],
            [half_size, -half_size, marker_size]
        ])

        color = (255, 0, 0)
        for rvec, tvec in zip(rvecs, tvecs):
            imgpts, jac = cv2.projectPoints(
                axis, rvec, tvec, mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            sides = frame.copy()
            sides = cv2.drawContours(sides, [imgpts[:4]], -1, color, -2)
            sides = cv2.drawContours(sides, [imgpts[4:]], -1, color, -2)
            sides = cv2.drawContours(
                sides, [np.array([imgpts[0], imgpts[1], imgpts[5], imgpts[4]])], -1, color, -2)
            sides = cv2.drawContours(
                sides, [np.array([imgpts[2], imgpts[3], imgpts[7], imgpts[6]])], -1, color, -2)
            sides = cv2.drawContours(
                sides, [np.array([imgpts[1], imgpts[2], imgpts[6], imgpts[5]])], -1, color, -2)
            sides = cv2.drawContours(
                sides, [np.array([imgpts[0], imgpts[3], imgpts[7], imgpts[4]])], -1, color, -2)
            frame = cv2.addWeighted(sides, 0.1, frame, 0.9, 0)
            frame = cv2.drawContours(frame, [imgpts[:4]], -1, color, 2)
            for ii, j in zip(range(4), range(4, 8)):
                frame = cv2.line(frame, tuple(
                    imgpts[ii]), tuple(imgpts[j]), color, 2)
            frame = cv2.drawContours(frame, [imgpts[4:]], -1, (255, 0, 0), 2)
        return frame

    def draw_points(self, frame, points, color):
        if len(points.shape) > 1:
            for p in points:
                cv2.circle(frame, tuple(p[:2].astype(int)),
                           3, color, -1)
        else:
            cv2.circle(frame, tuple(points[:2].astype(int)),
                       3, color, -1)

    def draw_text(self, frame, pos, texts):
        tl = 1.5
        tf = 1
        width = 0
        height = 0
        for text in texts:
            t_size = cv2.getTextSize(
                text, 0, fontScale=tl / 3, thickness=tf)[0]
            if t_size[0]>width:
                width = t_size[0]
            if t_size[1]>height:
                height = t_size[1]
        pad_height = 10
        c1 = tuple(
            np.array([pos[0] - width/2-2, pos[1] + height/2+pad_height/2]).astype(int))
        c2 = tuple(
            np.array([pos[0] + width/2+2, pos[1] - height/2-pad_height/2]).astype(int))
        for i, text in enumerate(texts):
            frame = cv2.rectangle(
                frame, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)  # filled
            frame = cv2.putText(frame, text, (c1[0], c1[1]-int(pad_height/2)), 0, tl / 3,
                                [225, 255, 255], thickness=tf,
                                lineType=cv2.LINE_AA)
            c1 = (c1[0], c1[1]+height+pad_height)
            c2 = (c2[0], c2[1]+height+pad_height)
        return frame
