import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage import data
from skimage.transform import resize

# Numpy is useful for handling arrays and matrices.
import numpy as np




#### Load Image
img = data.astronaut()
img = rgb2gray(img)*255 #convert to gray and change scale from (0,1) to (0,255).

n = img.shape[0]

plt.figure(1, figsize=(10, 10))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

#### Compute the differences operators here. Use your code from Assignment 1.
# You will need these three methods to construct sparse differences operators.
# If you do not use sparse operators you might have scalability problems.
from scipy.sparse import diags
from scipy.sparse import kron
from scipy.sparse import identity

# my import:
from scipy.sparse import csr_matrix # sparse matrix
from scipy.sparse.linalg import norm # compute the norm of a sparse matrix


# Use your code from Assignment 1. 
# Make sure that you compute the right D_h and D_v matrices.

m = img.shape[0] # ROWS
n = img.shape[1] # COLS
# plt.figure(0, figsize=(10, 10))
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.show()

# difference operator
diagsOfI = [1 for i in range(min(n, m))]
# print(diagsOfI)
I = diags([1], [0],shape=(n,n),dtype='int8').toarray()
# print('\nMatrix I -----')
# print(I)
J = diags([-1,1],[0,1],shape=(m,m),dtype='int8').toarray()
# print('\nMatrix j -----')
# print(J)
Dh = kron(J, I, format="csr")
Dv = kron(I, J, format="csr")
# print(D)


x = csr_matrix(np.reshape(img,(n*m,1))) 

Dh_x = Dh*x.toarray()
Dh_x = Dh_x.reshape(m,n)
# vertical Difference Operator: Done
Dv_x = Dv*x.toarray()
Dv_x = Dv_x.reshape(m,n)

# Add noise to the image
mean_ = 0
standard_deviation = 30
dimensions = (n,n)

noise = np.random.normal(mean_,standard_deviation,dimensions)
z_noise = csr_matrix(np.reshape(noisy_image, (n*m, 1)))

noisy_image = img + noise

plt.figure(1, figsize=(10, 10))
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
plt.show()

#### 
# Question 1: implement gradient descent using the Lipschitz constant as the 
# step-size for the denoising problem. Use eigsh method from scipy.sparse.linalg to compute the Lipschitz constant. Marks: 10
####
def gradient_descent(x0, epsilon, lambda_, max_iterations):
    
# x0: is the initial guess for the x variables
# epsilon: is the termination tolerance parameter
# lambda_: is the regularization parameter of the denoising problem.
# max_iterations: is the maximum number of iterations that you allow the algorithm to run.
# Write your code here.

    # make the D*D (conjugate transpost D multiply D) = DhT mul Dh + DvT mul Dv
    D_star_D = Dh.transpose()*Dh +Dv.transpose()*Dv 

    # stop when square of l2 norm of delta_f < epslion or reach max iteration
    delta_f = lambda_ / 2 * (norm(D_star_D))**2 + 1/2 *norm(x - z_noise)
    # while c < 2000 or 
