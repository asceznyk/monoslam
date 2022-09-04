import numpy as np

from utils import FundamentalMatrixTF

ftf = FundamentalMatrixTF()
src = np.random.rand(8,2)
dst = np.random.rand(8,2)
ftf.estimate(src, dst)

print(ftf.residuals(src, dst))


