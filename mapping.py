import numpy as np

class Map:
    def __init__(self):
        self.frames = [] 
        self.points = []

    def add_frame(self, frame):
        self.frames.append(frame)
        return len(self.frames)-1

    def add_point(self, point):
        self.points.append(point)
        return len(self.points)-1


