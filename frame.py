import cv2
import numpy as np

from skimage.measure import ransac
from scipy.spatial import cKDTree

from constants import *
from utils import EssentialMatrixTransform, calc_rt, normalize, add_ones 

def ext_features(img, max_corners=3000):
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), max_corners, qualityLevel=0.01, minDistance=7)
    kps, des = orb.compute(img, [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=20) for p in pts])
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    good_pts = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            p1, u1 = f1.kps[m.queryIdx], f1.kppx[m.queryIdx]
            p2, u2 = f2.kps[m.trainIdx], f2.kppx[m.trainIdx]

            if m.distance < 32 and m.queryIdx not in idx1 and m.trainIdx not in idx2:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                good_pts.append((p1, p2, u1, u2))

    assert(len(good_pts) >= RANSAC_MIN_SAMPLES)
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))

    good_pts = np.array(good_pts)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    model, inliers = ransac(
        (good_pts[:, 0], good_pts[:, 1]),
        EssentialMatrixTransform,
        min_samples=RANSAC_MIN_SAMPLES,
        residual_threshold=RANSAC_RES_THRESH,
        max_trials=RANSAC_MAX_TRIALS
    )

    return idx1[inliers], idx2[inliers], calc_rt(model.params, f1.K, f2.K, good_pts[inliers][:,2], good_pts[inliers][:,3])

class Frame:
    def __init__(self, graph, img, K=None, pose=np.eye(4), fid=None):
        self.K = np.array(K) if K is not None else K
        self.pose = np.array(pose)
        self.img = np.array(img)
        self.graph = graph

        self.id = fid if fid is not None else graph.add_frame(self)

        if self.img is not None:
            self.kppx, self.des = ext_features(self.img)

        self.pts = [None] * len(self.kppx)

    @property
    def KI(self):
        if not hasattr(self, '_KI'):
            self._KI = np.linalg.inv(self.K)
        return self._KI

    @property
    def kps(self):
        if not hasattr(self, '_kps'):
            self._kps = normalize(self.KI, self.kppx)
        return self._kps

    @property
    def kd(self):
        if not hasattr(self, '_kd'):
            self._kd = cKDTree(self.kppx)
        return self._kd




