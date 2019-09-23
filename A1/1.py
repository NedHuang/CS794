# load modules
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage import data
from skimage.transform import resize

# Numpy is useful for handling arrays and dense matrices (a matrix with a lot of nonzeros).
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse import kron
from scipy.sparse import identity
# load image
img = data.stereo_motorcycle()[0]
img = rgb2gray(img)*255 # convert to gray and change scale from (0,1) to (0,255).

m = img.shape[0] # ROWS
n = img.shape[1] # COLS

plt.figure(1, figsize=(10, 10))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

# m = 3
# n = 2

# difference operator
''' 
# this works for small matrix. use sparse now
j = np.zeros(shape=(m,n))
for a in range(min(j.shape[0],j.shape[1]-1)):
    b = a+1
    j[a][b] = 1
np.fill_diagonal(j,-1)

print('j is -----------\n')
print(j)

print('i is -----------\n')
i = np.zeros(shape=(n,m))
np.fill_diagonal(i,1)
print(i)
print('----------------')

D = np.zeros(shape=(m*n,n*m))
for a in range(j.shape[0]):
    for b in range(j.shape[1]):
        D[a*i.shape[0]:(a+1)*i.shape[0], b*i.shape[1]:(b+1)*i.shape[1]] = j[a][b]*i

print(D)

j = np.zeros(shape=(m,n),dtype=int)
for a in range(min(j.shape[0],j.shape[1]-1)):
    b = a+1
    j[a][b] = 1
np.fill_diagonal(j,-1)
print(j)
print('----------------')


i = np.zeros(shape=(n,m),dtype=int)
np.fill_diagonal(i,1)
print(i)

# D = np.zeros(shape=(m*n,n*m),dtype=int)
# for a in range(j.shape[0]):
#     for b in range(j.shape[1]):
#         D[a*i.shape[0]:(a+1)*i.shape[0], b*i.shape[1]:(b+1)*i.shape[1]] = j[a][b]*i
# D.astype(int)
# print(D)

print('----------------')

D = coo_matrix((m*n, m*n), dtype=np.int8).toarray()
row = []
col = []
data = []

for k in range(min(j.shape[0],j.shape[1])):
    for l in range(min(j.shape[0],j.shape[1])):
        row.append(k*i.shape[0]+l) 
        col.append(k*i.shape[1]+l)
        data.append(-1)
        if k*i.shape[1]+i.shape[1]+l<m*n:
            row.append(k*i.shape[0]+l) 
            col.append((k)*i.shape[0]+i.shape[1]+l)
            data.append(1)
D= coo_matrix((data, (row, col)), shape=(6, 6)).toarray()
print(D)
'''
diagsOfI = [1 for i in range(min(n, m))]
# print(diagsOfI)
I = diags(diagsOfI, 0,shape=(n,m),dtype='int8').toarray()
# print(I)
J = diags([-1,1],[0,1],shape=(m,n),dtype='int8').toarray()
# print(J)
D = kron(J, I, format="csr")
print(D)