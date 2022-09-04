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

class FundamentalMatrixTransform(object):
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




