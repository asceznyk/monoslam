import numpy as np

class Frame:
    def __init__(self, img, mapp):
        self.img = img
        self.mapp = mapp

        mapp.add_frame(self)
        self.id = len(mapp.frames)-1 


