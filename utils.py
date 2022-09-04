import cv2
import numpy as np

def add_ones(x):
    if len(x.shape) == 1: 
        return np.concatenate([x, np.array([1.0])], axis=0)
    return np.column_stack([x, np.ones(x.shape[0])])

def normalize(KI, pts):
    return (KI @ add_ones(pts).T).T[:, :2]

def in_front_of_both_cameras(pts1, pts2, R, t):
    RI = R
    for first, second in zip(pts1, pts2):
        first_z = (R[0]-second[0]*R[2]) @ t / (R[0]-second[0]*R[2]) @ second
        first_3d_point = np.array([first[0]*first_z, second[0]*first_z, first_z]) 
        second_3d_point = (R.T @ first_3d_point) - (R.T @ t)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True

def pose_rt(R, t):
    r_pose = np.eye(4)
    r_pose[:3, :3] = R
    r_pose[:3, 3] = t
    return r_pose

def skew_mat(x):
    return np.array([[0,-x[3],x[2]], [x[3],0,-x[1]], [-x[2],x[1],0]])

def decompose_essential_matrix(E, pts1, pts2):
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    U, S, VT = np.linalg.svd(E)
    
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(VT) < 0:
        VT *= -1.0

    R = U @ W @ VT
    if np.sum(R.diagonal()) < 0:
        R = U @ W.T @ VT

    t = U[:, 2]
    if t[2] < 0:
        t *= -1.0

    return np.linalg.inv(pose_rt(R, t))

class EssentialMatrixTransform(object):
    def __init__(self):
        self.params = np.eye(3)

    def __call__(self, coords):
        return np.column_stack([coords, np.ones(coords.shape[0])]) @ self.params.T

    def estimate(self, src, dst):
        assert src.shape == dst.shape
        assert src.shape[0] >= 8

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
            f_src[0]**2 + f_src[1]**2 + ft_dst[0]**2+ ft_dst[1]**2
        )




