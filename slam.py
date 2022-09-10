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

        self.graph = Map() 

    def process_frame(self, img):
        frame = Frame(self.graph, img, self.K)
        if frame.id == 0:
            return

        f1 = self.graph.frames[-1]
        f2 = self.graph.frames[-2] 

        idx1, idx2, pose = match_frames(f1, f2)
        f1.pose = f2.pose @ pose

        for i, idx in enumerate(idx2):
            if f1.pts[idx1[i]] is None and f2.pts[idx] is not None: 
                f2.pts[idx].add(f1, idx1[i])

        if len(self.graph.points) > 0:
            map_points = [add_ones(p) for p in self.graph.points]
            pts2d = ((self.K @ f1.pose[:3]) @ map_points.T).T
            pts2d = pts2d[:,:2] / pts2d[:,2:]

            good_pts =  (pts2d[:,0] > 0) & (pts2d[:,0] < self.W) & \
                        (pts2d[:,1] > 0) & (pts2d[:,1] < self.H)


        print(f1.des.shape)

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

        disp_map.paint(slam.graph)

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



