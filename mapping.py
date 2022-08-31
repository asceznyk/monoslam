import numpy as np

class Map:
    def __init__(self):
        self.frames = [] 
        self.points = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_point(self, point):
        self.points.append(point)


