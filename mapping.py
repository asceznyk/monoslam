import numpy as np

class Point:
    def __init__(self, graph, loc, pid=None):
        self.graph = graph
        self.loc = np.array(loc)
        self.frames = []
        self.idxs = []
        self.id = pid if pid is not None else graph.add_point(self)

    def orb(self):
        return [f.des[i] for f, i in zip(self.frame, self.idxs)]

    def orb_dist(self, des):
        return min([hamming_distance(o, des) for o in self.orb()])

    def add(self, frame, idx):
        assert frame not in self.frames
        assert frame.pts[idx] is None
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)

    def delete(self):
        for f, i in zip(self.frames, self.idxs):
            f.pts[i] = None
        del self

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


