import cv2
import numpy as np

def init_cam_intrinsics(img, f):
    w, h = img.shape[1], img.shape[0]
    P = np.array([
        [f, 0, w//2, 0],
        [0, f, h//2, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)
    return P, P[:3, :3]



