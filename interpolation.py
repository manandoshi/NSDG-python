import numpy as np


def lagrangian(x, X):
    N = X.size
    
    X = X + np.zeros([N, N])
    x = x + np.zeros([N, N])

    X_diff = X.T - X
    x_diff = x - X
    np.fill_diagonal(X_diff,1)
    np.fill_diagonal(x_diff,1)

    temp = x_diff/X_diff
    return np.prod(temp,1)

# ONLY FOR EVALUATION AT A NODE POINT
def dlagrangian(i, X):
    x = X[i]
    N = X.size
    
    X = X + np.zeros([N, N])
    x = x + np.zeros([N, N])

    X_diff = X.T - X
    x_diff = x - X
    np.fill_diagonal(X_diff,1)
    np.fill_diagonal(x_diff,1)
    x_diff = x_diff + (x_diff==0).astype(int)

    temp   = x_diff/X_diff
    diff_i = 1.0/X_diff[i,:]
    
    ret = np.prod(temp,1)
    ret[i] = np.sum(diff_i)-1
    return ret

def compute_matrix(X, X_outp):
    N = X.size
    M = X_outp.size

################################################
################################################
##    X_outp = np.reshape(X_outp, [1,1,-1])   ##
##    X_outp = X_outp + np.zeros([N,N,1])     ##
##                                            ##
##    X = X + np.zeros([N, N])                ##
##    np.fill_diagonal(X,0)                   ##
##    X_diff = X.T - X                        ##
##    np.fill_diagonal(X_diff,1)              ##
##    X_diff = X_diff + np.zeros([N,N,M])     ##
##                                            ##
##    x_diff = X_outp - np.reshape(X,[N,N,1]) ##
################################################
################################################

    matrix = np.zeros([M,N])
    for i,x in enumerate(X_outp):
        matrix[i,:] = lagrangian(x,X)
    
    return matrix

def compute_2Dmatrix(X, Y, outp_x, outp_y):
    
    nx,ny,Nx,Ny = X.size,Y.size,outp_x.size,outp_y.size

    mat_x = compute_matrix(X,outp_x)
    mat_y = compute_matrix(Y,outp_y)
    mat_x = np.concatenate([mat_x for i in xrange(ny)],axis=1)
    mat_x = np.concatenate([mat_x for i in xrange(Ny)],axis=0)
    mat_y = np.repeat(np.repeat(mat_y,nx,axis=1),Nx, axis=0)

    return mat_x*mat_y
