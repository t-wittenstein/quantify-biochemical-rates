#!/usr/bin/python

import numpy as np
from cvxopt import solvers, matrix, spmatrix, mul
import itertools
from scipy import sparse

#Help functions:

def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
    return SP

def spmatrix_sparse_to_scipy(A):
    data = np.array(A.V).squeeze()
    rows = np.array(A.I).squeeze()
    cols = np.array(A.J).squeeze()
    return sparse.coo_matrix( (data, (rows, cols)) )

def sparse_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return sparse.vstack([A1, A2])

def numpy_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.vstack([A1, A2])
        
def numpy_None_concatenate(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.concatenate([A1, A2])

def get_shape(A):
    if isinstance(C, spmatrix):
        return C.size
    else:
        return C.shape

def numpy_to_cvxopt_matrix(A):
    if A is None:
        return A
    if sparse.issparse(A):
        if isinstance(A, sparse.spmatrix):
            return scipy_sparse_to_spmatrix(A)
        else:
            return A
    else:
        if isinstance(A, np.ndarray):
            if A.ndim == 1:
                return matrix(A, (A.shape[0], 1), 'd')
            else:
                return matrix(A, A.shape, 'd')
        else:
            return A

def cvxopt_to_numpy_matrix(A):
    if A is None:
        return A
    if isinstance(A, spmatrix):
        return spmatrix_sparse_to_scipy(A)
    elif isinstance(A, matrix):
        return np.array(A).squeeze()
    else:
        return np.array(A).squeeze()
        

        
#Optimization Algorithm 

def lsqlin(C, d, reg=0, A=None, b=None, Aeq=None, beq=None, \
        lb=None, ub=None, x0=None, opts=None):
    '''
        Solve:
            min_x ||C*x  - d||^2_2 + reg * ||D*x||^2_2
            s.t.  A * x <= b
                  Aeq * x = beq
                  lb <= x <= ub

        Input arguments:
            C   is m x n dense or sparse matrix
            d   is n x 1 dense matrix
            reg is regularization parameter
            A   is p x n dense or sparse matrix
            b   is p x 1 dense matrix
            Aeq is q x n dense or sparse matrix
            beq is q x 1 dense matrix
            lb  is n x 1 matrix or scalar
            ub  is n x 1 matrix or scalar
            
        Regularization matrix defined below::
            D   is m-2 x n regularization matrix

    '''
    sparse_case = False
    if sparse.issparse(A): #detects both np and cxopt sparse
        sparse_case = True
        if isinstance(A, spmatrix):
            A = spmatrix_sparse_to_scipy(A)
            
    C =   numpy_to_cvxopt_matrix(C)
    d =   numpy_to_cvxopt_matrix(d)
    Q = C.T * C
    q = - d.T * C
    nvars = C.size[1]

    # Adding Regularization as an approximation of the second order derivative
    if reg > 0:
        
        #Regularization Matrix
        Dreg = [] 
        for i in range(nvars-2):
            Dreg.append(i*[0]+[1]+[-2]+[1]+(nvars-3-i)*[0])
        Dreg = matrix(np.array(Dreg))
        
        if sparse_case:
            I = scipy_sparse_to_spmatrix(sparse.eye(nvars, nvars,\
                                          format='coo'))
        else:
            I = matrix(np.eye(nvars), (nvars, nvars), 'd')
        Q = Q + reg * Dreg.T * Dreg

    lb = cvxopt_to_numpy_matrix(lb)
    ub = cvxopt_to_numpy_matrix(ub)
    b  = cvxopt_to_numpy_matrix(b)
    
    if lb is not None:  #Modify 'A' and 'b' to add lb inequalities 
        if lb.size == 1:
            lb = np.repeat(lb, nvars)
    
        if sparse_case:
            lb_A = -sparse.eye(nvars, nvars, format='coo')
            A = sparse_None_vstack(A, lb_A)
        else:
            lb_A = -np.eye(nvars)
            A = numpy_None_vstack(A, lb_A)
        b = numpy_None_concatenate(b, -lb)
    if ub is not None:  #Modify 'A' and 'b' to add ub inequalities
        if ub.size == 1:
            ub = np.repeat(ub, nvars)
        if sparse_case:
            ub_A = sparse.eye(nvars, nvars, format='coo')
            A = sparse_None_vstack(A, ub_A)
        else:
            ub_A = np.eye(nvars)
            A = numpy_None_vstack(A, ub_A)
        b = numpy_None_concatenate(b, ub)

    #Convert data to CVXOPT format
    A =   numpy_to_cvxopt_matrix(A)
    Aeq = numpy_to_cvxopt_matrix(Aeq)
    b =   numpy_to_cvxopt_matrix(b)
    beq = numpy_to_cvxopt_matrix(beq)

    #Set up options
    if opts is not None:
        for k, v in opts.items():
            solvers.options[k] = v
            
    solvers.options['show_progress'] = False #comment out to see optimization progress
    
    #Run CVXOPT.SQP solver
    sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, x0)
    return sol

#Function infering reaction rates, needs joint probability distribution and regularization parameter as arguments

def Rate_Inference_main(P_joint, reg_param, tau = 1):

    P_joint = P_joint/sum(sum(P_joint)) #Normalize distribution

    P_upstream = sum(P_joint.T) #Upstream variable distribution
    P_downstream = sum(P_joint) #Downstream variable distribution


    Gij = np.zeros((len(P_downstream)-1,len(P_upstream))) #Independent variable Matrix
    for j in range(len(Gij)):
        if P_downstream[j] != 0:
            Gij[j] = P_joint.T[:-1][j]
        else:
            Gij[j] = P_joint.T[:-1][j]
            
    Bj = np.zeros(len(P_downstream)-1) #Dependent variable Vector
    for j in range(len(Bj)):
        Bj[j] = (j+1)*P_downstream[j+1]/tau

    #Infering Reaction Rate
    f_est = lsqlin(Gij, Bj, reg = reg_param, lb = np.zeros(len(P_upstream)))['x']
    f_est = np.array([i for i in f_est])
    
    return f_est

#Adding monotonicity

def Rate_Inference_Monotonicity(P_joint, reg_param, tau = 1, mon = 'pos'):

    P_joint = P_joint/sum(sum(P_joint))

    P_upstream = sum(P_joint.T)
    P_downstream = sum(P_joint)


    Gij = np.zeros((len(P_downstream)-1,len(P_upstream))) #Independent variable Matrix
    for j in range(len(Gij)):
        if P_downstream[j] != 0:
            Gij[j] = P_joint.T[:-1][j]
        else:
            Gij[j] = P_joint.T[:-1][j]
            
    Bj = np.zeros(len(P_downstream)-1) #Dependent variable Vector
    for j in range(len(Bj)):
        Bj[j] = (j+1)*P_downstream[j+1]/tau
        
    #Adding Monotonicity constraint
    L = []
    for i in range(len(P_upstream)-1):
        L.append(i*[0]+[-1]+[1]+(len(P_upstream)-2-i)*[0])
    if mon == 'pos':
        L = -np.array(L)
    elif mon == 'neg':
        L = np.array(L)  
    else:
        print('Please define the monotonicity as either "pos" or "neg"')
        return
    h = np.zeros(len(P_upstream)-1)

    #Infering Reaction Rate
    f_est = lsqlin(Gij, Bj, reg = reg_param, A=L, b=h, lb = np.zeros(len(P_upstream)))['x']
    f_est = np.array([i for i in f_est])
    
    return f_est

#Code for different system where the variable degrades via dimerization

def Rate_Inference_Dimerization(P_joint, reg_param, gamma = 1):

    P_joint = P_joint/sum(sum(P_joint)) #Normalize distribution

    P_upstream = sum(P_joint.T) #Upstream variable distribution
    P_downstream = sum(P_joint) #Downstream variable distribution


    Gij = np.zeros((len(P_downstream)-1,len(P_upstream))) #Independent variable Matrix
    for j in range(len(Gij)):
        if P_downstream[j] != 0:
            Gij[j] = P_joint.T[:-1][j]
        else:
            Gij[j] = P_joint.T[:-1][j]
            
    Bj = np.zeros(len(P_downstream)-1) #Dependent variable Vector
    for j in range(len(Bj)):
        Bj[j] = gamma*(j+1)*(j+2)*P2[j+2] + gamma*(j+1)*(j+0)*P2[j+1]

    #Infering Reaction Rate
    f_est = lsqlin(Gij, Bj, reg = reg_param, lb = np.zeros(len(P_upstream)))['x']
    f_est = np.array([i for i in f_est])
    
    return f_est