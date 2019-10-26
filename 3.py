import copy
from numpy.linalg import norm
import math
D = Dh + 1J * Dv
epsilon = 1.0e-2
max_iterations = 2000
lambda_ = 5
miu_ = 1
x0 = noisy_image

def gradient_descent(x0, epsilon, lambda_, max_iterations,gamma_, miu_):
    # initialize variables
    counter = 0
    x = x0
    xs = list()
    xs.append(x)

    
    Dx = D.dot(x)
    phi_Dx =  sum( ((miu_**2 - dxi**2)**(1/2)-miu_) for dxi in Dx)

    gradient_fx = calculate_gradient_fx(x,miu_)
    while counter < max_iterations and norm(gradient_fx,2) > epsilon:
        print(counter)
        alpha_ = line_search_Armijo(x, gradient_fx, gamma_)
        x = x - alpha_ * gradient_fx
        gradient_fx = calculate_gradient_fx(x)
        xs.append(x)
        counter += 1
    return

# helper function, take x, return gradient of f(x)
def calculate_gradient_fx(x,miu_):
    gradient_fx = copy.deepcopy(x)
    # calculate the gradient of φ_μ(Dx) first, then gradient of f(x)
    for xi in gradient_fx:
        xi = xi*(1+(xi**2/miu_**2))**(-1/2)
    gradient_fx += x - x0
    return gradient_fx/miu_


def line_search_Armijo(x, grad, gamma_):
    # initial guess
    alpha_ = 1
    diff = x - alpha_ * grad
    # store the value for reuse
    deno_x = denoising(x)
    LHS = denoising(diff)
    RHS = deno_x - alpha_ * gamma_ * norm(grad) ** 2
    print('RHS', RHS)
    while LHS > RHS:
        alpha_ /= 2
        print('alpha_ ', alpha_)
        diff = x - alpha_ * grad
        # LHS = denoising(diff)
        RHS = deno_x - alpha_ * gamma_ * norm(grad) ** 2
    return alpha_

def denoising(x):
    Dx = D.dot(x)
    return lambda_ * sum( ((miu_**2 - dxi**2)**(1/2)-miu_) for dxi in Dx) + 1/2 * math.pow(norm(x-x0),2)

# lambda_ = 4
# epsilon = 1.0e-2
# max_iterations = 2000
# gamma_ =0.4
# miu_ = 1
# D = Dh + 1J * Dv

# gradient_descent(x0, epsilon, lambda_,max_iterations,gamma_,miu_)

x = x0
print(denoising(x))


gradient_descent(x0, epsilon, lambda_, max_iterations,gamma_, miu_)