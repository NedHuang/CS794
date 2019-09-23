# load modules
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage import data
from skimage.transform import resize

# Numpy is useful for handling arrays and dense matrices (a matrix with a lot of nonzeros).
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
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
# plt.show()


# difference operator

diagsOfI = [1 for i in range(min(n, m))]

# print(diagsOfI)
I = diags([1], [0],shape=(n,n),dtype='int8').toarray()
print('\nMatrix I -----')
print(I)
J = diags([-1,1],[0,1],shape=(m,m),dtype='int8').toarray()
print('\nMatrix j -----')
print(J)
Dh = kron(J, I, format="csr")
Dv = kron(I, J, format="csr")
# print(D)

# problem 2
x = csr_matrix(np.reshape(img,(n*m,1))) 

Dh_x = Dh*x.toarray()
Dh_x = Dh_x.reshape(m,n)
# vertical Difference Operator: Done
Dv_x = Dv*x.toarray()
Dv_x = Dv_x.reshape(m,n)

plt.figure(1, figsize=(10, 10))
plt.imshow(Dh_x, cmap='gray', vmin=0, vmax=255)
plt.show()

'''

mean_ = 0
standard_deviation = 30
dimensions = (m,n)

noise = np.random.normal(mean_,standard_deviation,dimensions)

noisy_image = img + noise

plt.figure(1, figsize=(10, 10))
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
plt.show()
'''