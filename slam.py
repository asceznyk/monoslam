import os
import cv2
import argparse

import numpy as np

from tqdm import tqdm

from utils import *
from frame import * 
from mapping import *

class SLAM:
    def __init__(self, focal_length):
        self.focal_length = focal_length
        self.mapp = Map() ## this has frames

    def process_frame(self, img):
        frame = Frame(img, self.mapp) 
        if frame.id == 0: 
            self.P, self.K = init_cam_intrinsics(img, self.focal_length)
            print(self.P)
            return

def main(video_path, focal_length=910):
    cap = cv2.VideoCapture(video_path)
    ret = True

    slam = SLAM(focal_length)
    cur_pose = np.eye(4)

    f = 0
    while ret:
        ret, img = cap.read()
        if ret:
            slam.process_frame(img)
        f += 1 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='path to the video you want to track')
    parser.add_argument('--focal_length', type=int, default=716, help='focal length of the camera for building projection matrix if calibration path is not given')

    args = parser.parse_args()
    print(args)

    list_args = []
    for k in args.__dict__:
        list_args.append(args.__dict__[k])
    main(*list_args)



