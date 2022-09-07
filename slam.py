#!/usr/bin/env python3

import os
import cv2
import argparse

import numpy as np

from tqdm import tqdm

from utils import *
from frame import * 
from mapping import *
from display import *

class SLAM:
    def __init__(self, W, H, K):
        self.W = W
        self.H = H
        self.K = K

        print(self.K)

        self.mapp = Map() 

    def process_frame(self, img):
        frame = Frame(self.mapp, img, self.K)
        if frame.id == 0:
            return

        f1 = self.mapp.frames[-1]
        f2 = self.mapp.frames[-2] 

        idx1, idx2, pose = match_frames(f1, f2)
        f1.pose = f2.pose @ pose

def main(video_path, focal_length=524):
    cap = cv2.VideoCapture(video_path)
    ret = True

    F = focal_length
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'actual img dim {W}x{H} and focal_length {F}')

    if W > 1024.0:
        d = 1024.0/W
        F *= d
        H = int(H*d)
        W = 1024

    print(f'init cam focal_length using {F} and dim {W}x{H}..')

    K = np.array([[F,0,W//2], [0,F,H//2], [0,0,1]])

    slam = SLAM(W, H, K)
    disp_map = DisplayMap()
    cur_pose = np.eye(4)

    f = 0
    while ret:
        ret, img = cap.read()
        img = cv2.resize(img, (W, H))
        if ret:
            print(f'{f}/{count}')
            slam.process_frame(img)
        f += 1

        disp_map.paint(slam.mapp)

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



