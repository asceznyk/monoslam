import cv2
import numpy as np

from skimage.measure import ransac

class Frame:
    def __init__(self, mapp, img, K=None, pose=np.eye(4)):
        self.K = np.array(K) if self.K is not None else K
        self.pose = np.array(pose)

        self.img = np.array(img)
        self.mapp = mapp

        mapp.add_frame(self)
        self.id = len(mapp.frames)-1 

        if self.img is not None:
            self.kps, self.des = ext_features(self.img)


