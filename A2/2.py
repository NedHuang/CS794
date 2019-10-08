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
# plt.show()

#### Compute the differences operators here. Use your code from Assignment 1.
# You will need these three methods to construct sparse differences operators.
# If you do not use sparse operators you might have scalability problems.
from scipy.sparse import diags
from scipy.sparse import kron
from scipy.sparse import identity

# my import:
from scipy.sparse import csr_matrix # sparse matrix
from scipy.sparse.linalg import norm # compute the norm of a sparse matrix
from scipy import real
from scipy.sparse.linalg import spsolve 
from scipy.sparse.linalg import eigsh 


import math
from numpy.linalg import norm # for not sparse matrix/vector


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
# Dh = kron(J, I, format="csr")
# Dv = kron(I, J, format="csr")

Dh = kron(J, I)
Dv = kron(I, J)
# print(D)


# x = csr_matrix(np.reshape(img,(n*m,1))) 

# Dh_x = Dh*x.toarray()
# Dh_x = Dh_x.reshape(m,n)
# # vertical Difference Operator: Done
# Dv_x = Dv*x.toarray()
# Dv_x = Dv_x.reshape(m,n)

# Add noise to the image
mean_ = 0
standard_deviation = 30
dimensions = (n,n)

noise = np.random.normal(mean_,standard_deviation,dimensions)
noisy_image = img + noise
# vectorize
z_noise = csr_matrix(np.reshape(noisy_image, (n*m, 1)))

plt.figure(1, figsize=(10, 10))
plt.imshow(noisy_image, cmap='gray', vmin=0, vmax=255)
# plt.show()


###############################################################################
# Question 1
###############################################################################

lambda_ = 4
epsilon = 1.0e-2
max_iterations = 2000

# find lipschitz constant 
# !!! we comment this because computing L is time-consuming.
# I_mn_mn = identity(m*n)
# A = lambda_*(Dh.transpose().dot(Dh) + Dv.transpose().dot(Dv)) + I_mn_mn
# eigv = eigsh(A.transpose().dot(A), 1, which='LM', return_eigenvectors=False)
# L = math.sqrt(eigv)
# eigv =  1088.98015996, L = 32.999699391964164
L = 32.999699391964164


def gradient_descent(x0, epsilon, lambda_, max_iterations):
# x0: is the initial guess for the x variables
# epsilon: is the termination tolerance parameter
# lambda_: is the regularization parameter of the denoising problem.
# max_iterations: is the maximum number of iterations that you allow the algorithm to run.
# x_k+1 = x_k -1/L ×gradient(x_k)

    counter = 0
    x = x0
    xs = list()     # xs is list of the x calcualted in each iteration
    xs.append(x)
    D = Dh + 1J * Dv
    # D_conjugate_transpose = D.conjugate().transpose()
    # lecture 5 Slide 31
    Dct = D.conjugate().transpose()


    gradient_f_x = lambda_ * real(Dct.dot(D.dot(x))) + x - x0  # x0 is z_noisy
    gradient_f_x = gradient_f_x.toarray()                       # need to convert to ndarray. Otherwise wont work...
    while counter < max_iterations and norm(gradient_f_x,2) > epsilon:
        x = x - (1/L)*gradient_f_x              # lecture 5 slide 16, cumputer x_k+1
        xs.append(x.flatten('F'))
        gradient_f_x = lambda_ * real(Dct.dot(D.dot(x))) + x - x0   # update gradient
        counter += 1                                                # update countertoarray
    print('finished gradient descent')
    return xs

###############################################################################
# Call Gradient Descent
###############################################################################



# x0 = np.reshape(noisy_image, (n*n, 1)) 
# !  this will Unable to allocate array with shape (262144, 262144) and data type float64 
x0 = noisy_image.flatten('F')
print(1,type(x0),x0.shape)

list_of_x = gradient_descent(z_noise, epsilon, lambda_, max_iterations)
print(2,type(list_of_x[-1]),list_of_x[-1].shape)
D = Dh + 1J * Dv



# calculate A and solve for x by spsolve from Assignment 1, lecture 3, slide 24
A = lambda_*real(Dh.transpose().dot(Dh) + Dv.transpose().dot(Dv)) + identity(m*n)
x_minimizer = spsolve(A,x0)  
# # denoising = lambda x : lambda_ / 2 * norm(D.dot(x), 2) ** 2 + 1/2 * norm(x-x0) ** 2


# helper function, denoising, solve the x
def denoising(x):
    if(type(x) =='scipy.sparse.csr.csr_matrix'):
        x = x.toarray()

    return lambda_/2 * math.pow(norm(D.dot(x), 2),2) + 1/2 * math.pow(norm(x-x0),2)

# # calculate f(x*), f(x0) and plot 
# f(xk) - f(x*)
# —————————————
# f(x0) - f(x*)
f_minimizer = denoising(x_minimizer)
f_x0 = denoising(x0)
denominator = f_x0-f_minimizer
print(type(list_of_x[0]), list_of_x[0].shape,list_of_x[0].dtype)
print(type(x0), x0.shape,x0.dtype)
# modify the example code
store_data_for_plotting = []
for x in list_of_x[1:]:
    nume = denoising(x.flatten('F')) - f_minimizer
    # store ratio
    store_data_for_plotting.append(nume / denominator)
# fig = plt.figure(figsize=(8, 6))
# plt.plot(store_data_for_plotting, label=("Gradient descent + Lipschitz"), linewidth=5.0, color ="black")

# plt.legend(prop={'size': 20},loc="upper right")
# plt.xlabel("iteration $k$", fontsize=25)
# plt.ylabel("Rellative distance to opt.", fontsize=25)
# plt.grid(linestyle='dashed')
# plt.show()
