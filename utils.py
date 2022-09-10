import cv2
import numpy as np

from constants import *

def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero(((a ^ b) & r) != 0)

def add_ones(x):
    if len(x.shape) == 1: 
        return np.concatenate([x, np.array([1.0])], axis=0)
    return np.column_stack([x, np.ones(x.shape[0])])

def normalize(KI, pts):
    return (KI @ add_ones(pts).T).T[:, :2]

def pose_rt(R, t):
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = t
    return P

def skew_mat(x):
    return np.array([[0,-x[2],x[1]], [x[2],0,-x[0]], [-x[1],x[0],0]])

def decompose_matrix(E):
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    U, S, VT = np.linalg.svd(E)
    if np.linalg.det(U) < 0: U *= -1.0
    if np.linalg.det(VT) < 0: VT *= -1.0
    R1, R2 = U @ W @ VT, U @ W.T @ VT
    return R1, R2, U[:,2]

def calc_rt(E, K1, K2, q1, q2):
    P1 = np.column_stack([K1, np.zeros((3,1))])
    P2 = np.column_stack([K2, np.zeros((3,1))])

    def sum_z_cal_relative_scale(R, t):
        pose = pose_rt(R, t)
        hom_q1 = cv2.triangulatePoints(P1, P2 @ pose, q1.T, q2.T) 
        hom_q2 = pose @ hom_q1
        uhom_q1 = hom_q1[:3, :] / (hom_q1[3, :] + 1e-24)
        uhom_q2 = hom_q2[:3, :] / (hom_q2[3, :] + 1e-24)
        return sum(uhom_q1[2,:] > 0) + sum(uhom_q2[2,:] > 0)

    R1, R2, t = decompose_matrix(E)
    t = np.squeeze(t)

    pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
    sumzs = []
    for i, [R, t] in enumerate(pairs): 
        sumzs.append((i, sum_z_cal_relative_scale(R, t)))

    j, _ = max(sumzs, key=lambda x: x[1])
    R, t = pairs[j]
    return pose_rt(R, t)

class EssentialMatrixTransform(object):
    def __init__(self):
        self.params = np.eye(3)

    def __call__(self, coords):
        return np.column_stack([coords, np.ones(coords.shape[0])]) @ self.params.T

    def estimate(self, src, dst):
        assert src.shape == dst.shape
        assert src.shape[0] >= RANSAC_MIN_SAMPLES

        A = np.ones((src.shape[0], 9))
        A[:, :2] *= src
        A[:, :3] *= dst[:, 0, np.newaxis]
        A[:, 3:5] *= src
        A[:, 3:6] *= dst[:, 1,  np.newaxis]
        A[:, 6:8] *= src

        _, _, VT = np.linalg.svd(A)
        F = VT[-1].reshape(3, 3) 

        U, S, VT = np.linalg.svd(F)
        S[0] = S[1] = (S[0] + S[1]) / 2.0
        S[2] = 0
        self.params = U @ np.diag(S) @ VT

        return True

    def residuals(self, src, dst):
        src_hom = np.column_stack([src, np.ones(src.shape[0])])
        dst_hom = np.column_stack([dst, np.ones(dst.shape[0])])

        f_src = self.params @ src_hom.T
        ft_dst = self.params.T @ dst_hom.T

        dst_f_src = np.sum(dst_hom * f_src.T, axis=1)

        return np.abs(dst_f_src) / np.sqrt(
            f_src[0]**2 + f_src[1]**2 + ft_dst[0]**2 + ft_dst[1]**2
        )




