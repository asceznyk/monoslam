import pangolin

import numpy as np
import OpenGL.GL as gl

from multiprocessing import Process, Queue

class DisplayMap:
    def __init__(self):
        self.state = None
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('MapView', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            pangolin.ModelViewLookAt(
                0, -10, -8,
                0, 0, 0,
                0, -1, 0
            )
        )
        self.handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
        self.dcam.SetHandler(self.handler)
        self.dcam.Activate()

    def viewer_refresh(self, q):
        while not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        if self.state is not None:
            if self.state[0].shape[0] >= 2:
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawCameras(self.state[0][:-1])

            if self.state[0].shape[0] >= 1:
                gl.glColor3f(0.0, 1.0, 0.0)
                pangolin.DrawCameras(self.state[0][-1:])

            if self.state[1].shape[0] != 0:
                gl.glPointSize(5)
                gl.glColor3f(1.0, 0.0, 0.0)
                pangolin.DrawPoints(self.state[1], self.state[2])

        pangolin.FinishFrame()

    def paint(self, graph):
        poses, points, colors = [], [], []
        for p in graph.points:
            points.append(p.pt)
            colors.append(p.color)
        for f in graph.frames:
            poses.append(np.linalg.inv(f.pose))

        self.q.put((np.array(poses), np.array(points), np.array(colors)/255.0))



