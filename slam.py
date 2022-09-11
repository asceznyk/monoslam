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

        idx1, idx2, P = match_frames(f1, f2)
        f1.pose = P @ f2.pose

        for i, idx in enumerate(idx2):
            if f1.pts[idx1[i]] is None and f2.pts[idx] is not None: 
                f2.pts[idx].add(f1, idx1[i])

        sbp_pts_cnt = 0
        if len(self.graph.points) > 0:
            map_points = np.array([p.homogenous() for p in self.graph.points])
            pts2d = ((self.K @ f1.pose[:3]) @ map_points.T).T
            pts2d = pts2d[:,:2] / pts2d[:,2:]

            good_pts = (pts2d[:,0] > 0) & (pts2d[:,0] < self.W) & \
                (pts2d[:,1] > 0) & (pts2d[:,1] < self.H)

            for i, p in enumerate(self.graph.points):
                if not good_pts[i]:
                    continue
                if f1 in p.frames:
                    continue

                for j in f1.kd.query_ball_point(pts2d[i], 2):
                    if f1.pts[j] is None:
                        if p.orb_distance(f1.des[j]) < 64.0:
                            p.add(f1, j)
                            sbp_pts_cnt += 1
                            break

        good_pts4d = np.array([f1.pts[i] is None for i in idx1])
        pts4d = triangulate_pts(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2]) 
        good_pts4d &= np.abs(pts4d[:, 3]) != 0
        pts4d /= pts4d[:, 3:] 

        new_pts_cnt = 0
        rejected_pts_cnt = 0
        for i, p in enumerate(pts4d):
            if not good_pts4d[i]:
                continue

            pp1 = f1.pose @ p
            pp2 = f2.pose @ p
            if pp1[2] < 0 or pp2[2] < 0:
                rejected_pts_cnt += 1
                continue

            rpp1 = self.K @ pp1[:3]
            rpp2 = self.K @ pp2[:3]
            rpp1 = np.sum(((rpp1[:2] / rpp1[2:]) - f1.kppx[idx1[i]])**2)
            rpp2 = np.sum(((rpp2[:2] / rpp2[2:]) - f2.kppx[idx2[i]])**2)
            if rpp1 > 2 or rpp2 > 2:
                rejected_pts_cnt += 1
                continue

            kppx = f1.kppx[idx1[i]] 
            pt = Point(self.graph, p[:3], img[int(round(kppx[1])), int(round(kppx[0]))])
            pt.add(f1, idx1[i])
            pt.add(f2, idx2[i])
            new_pts_cnt += 1

        print(f"Adding {new_pts_cnt} points - {sbp_pts_cnt} points searched")
        print(f"Rejected points: {rejected_pts_cnt}")
        print(f"Map points: {len(self.graph.points)} frames: {len(self.graph.frames)}")

        print(np.linalg.inv(f1.pose))

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



