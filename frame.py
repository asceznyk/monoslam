import cv2
import numpy as np

from skimage.measure import ransac

def ext_features(orb, img, max_corners=3000): 
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), max_corners, qualityLevel=0.01, minDistance=7)
    kps, des = orb.compute(img, [cv2.KeyPoint(p[0][0], p[0][1], size=20) for p in pts])
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

class Frame:
    def __init__(self, mapp, orb, img, K=None, pose=np.eye(4)):
        self.K = np.array(K) if K is not None else K
        self.pose = np.array(pose)
        self.img = np.array(img)
        self.mapp = mapp

        mapp.add_frame(self)
        self.id = len(mapp.frames)-1 

        if self.img is not None:
            self.kps, self.des = ext_features(orb, self.img)




