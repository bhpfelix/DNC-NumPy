# Imports
import autograd.numpy as np
from scipy.stats import truncnorm
from autograd import grad
import pickle as pk
import matplotlib.pyplot as plt

__author__ = 'Haoping Bai'
__copyright__ = 'Copyright (c) 2018, Haoping Bai'
__email__ = 'bhpfelix@gmail.com'
__license__ = 'MIT'

def load_pickle(filename):
    try:
        p = open(filename, 'rb')
    except IOError:
        print "Pickle file cannot be opened."
        return None
    try:
        picklelicious = pk.load(p)
    except ValueError:
        print 'load_pickle failed once, trying again'
        p.close()
        p = open(filename, 'rb')
        picklelicious = pk.load(p)

    p.close()
    return picklelicious

def save_pickle(data_object, filename):
    pickle_file = open(filename, 'wb')
    pk.dump(data_object, pickle_file)
    pickle_file.close()
    

# Utils: Numerically Stable, Vectorized implementations of util functions
def nprn(*size):
    return truncnorm.rvs(-2, 2, size=size)

def l2(x):
    """
    stable l2-norm
    """
    return np.sqrt(np.sum(np.multiply(x, x)) + 1.e-20) 

def oneplus(x, limit=30.):
    """
    Numerically stable implementation: | log(1+exp(30)) - 30 | < 1e-10
    Constraint to [1, inf)
    """
#     limit = 30
#     x[x < limit] = np.log(1. + np.exp(x[x < limit]))
    x = np.log(1. + np.exp(x))
    return 1. + x

def sigmoid(x):
    """
    Constraint to [0, 1]
    """
    return 1. / (1. + np.exp(-x))

def softmax(x): # row-wise softmax
    res = np.exp(x - np.max(x, axis=1, keepdims=True))
    res /= np.sum(res, axis=1, keepdims=True)
    return res

def shift_cumprod(x):
    # solve dimension problem
    c = np.squeeze(x)
    slices = [1.]
    for i in range(1,len(c)):
        slices.append(np.prod(c[:i]))
    return np.array([slices])

def cos_sim(u, v):
    """
    Cosine similarity between u,v
    """
    n = np.dot(u,v)
    d = np.sqrt(np.dot(u,u) * np.dot(v,v))
    d += 1.e-20 # prevent undefined cos similarity at 0 from breaking the code
    return n / d

def d_tanh(x):
    """
    Derivative for tanh used for gradient calculation
    """
    y = np.tanh(x)
    return 1. - y * y

def d_sigmoid(x):
    """
    Derivative for sigmoid used for gradient calculation
    """
    y = sigmoid(x)
    return y * (1. - y)

def d_cos_sim(u, v):
    """
    Differentiate Cos Similarity with regard to u, switch order to calculate for v
    """
    n = np.dot(u, v)
    u_inner_prod = np.dot(u*u)
    d = np.sqrt(u_inner_prod * np.dot(v*v)) + 1.e-20 # deal with undefined cos sim for zero vec

    return v / d - (n / d) * (u / u_inner_prod)    
    
    
def display(arr):
    plt.imshow(arr, cmap='Greys')
#     plt.yticks([])
#     plt.xticks([])
    plt.show()
