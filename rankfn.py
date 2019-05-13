import numpy as np

A = np.array([1,2,3,4,5,6,7,8,9])
pct = np.array(list(map(lambda x: np.count_nonzero(A <= x)/len(A), A)))
print(np.interp([1.5,2], A, pct))
print(pct)
