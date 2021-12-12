#!/usr/bin/python

import numpy as np


##Function to sample N independent points from a two dimensional distribution P_joint:

def Sample_Dist(P_joint,N):
    
    shape = np.shape(P_joint)
    P_samp = P_joint.reshape((1,shape[0]*shape[1]))[0]
    P_ind = np.arange(len(P_samp))

    Traj = np.random.choice(P_ind,size = N, p = P_samp)
    Pab_r = np.zeros(len(P_samp))
    for i in Traj:
        Pab_r[i] += 1
    Pab_r = Pab_r/sum(Pab_r)
    Pab_r = Pab_r.reshape(shape)

    return Pab_r




## Sampling functions applying different error models on the measured data, smaples N data points:

# Adding binominal error with a chance of probability p for each molecule to get detected
def Sample_Dist_error_bin(P_joint,N,p):
    
    shape = np.shape(P_joint)
    P_samp = P_joint.reshape((1,shape[0]*shape[1]))[0]

    P_ind = np.arange(len(P_samp))

    P_ind_tup = [[0 for i in range(shape[1])] for i in range(shape[0])]
    for i in range(len(P_ind_tup)):
        for j in range(len(P_ind_tup[0])):
            P_ind_tup[i][j] = (i,j)     
    P_ind_tup = np.array(P_ind_tup).reshape((1,shape[0]*shape[1],2))[0]

    Traj = np.random.choice(P_ind,size = N, p = P_samp)

    Traj_tup = []
    for i in Traj:
        cond = False
        while cond == False:
            rand_tup = (np.random.binomial(P_ind_tup[i][0],p),np.random.normal(P_ind_tup[i][1],p))
            if rand_tup[0] >= 0 and rand_tup[1] >= 0:
                Traj_tup.append([int(rand_tup[0]+0.5), 
                                 int(rand_tup[1]+0.5)])
                cond = True
    Traj_tup = np.array(Traj_tup)

    shape_new = (np.max(Traj_tup[:,0])+1,np.max(Traj_tup[:,1])+1)

    Pab_r = np.zeros((shape_new[0],shape_new[1]))
    for tup in Traj_tup:
        Pab_r[tup[0],tup[1]] += 1
    Pab_r = Pab_r/sum(sum(Pab_r))

    return Pab_r


# Adding an absolute error on each data point with a standard deviation of sig_abs, negative molecule numbers get ignored
def Sample_Dist_error_abs(P_joint,N,sig_abs):
    
    shape = np.shape(P_joint)
    P_samp = P_joint.reshape((1,shape[0]*shape[1]))[0]

    P_ind = np.arange(len(P_samp))

    P_ind_tup = [[0 for i in range(shape[1])] for i in range(shape[0])]
    for i in range(len(P_ind_tup)):
        for j in range(len(P_ind_tup[0])):
            P_ind_tup[i][j] = (i,j)     
    P_ind_tup = np.array(P_ind_tup).reshape((1,shape[0]*shape[1],2))[0]

    Traj = np.random.choice(P_ind,size = N, p = P_samp)

    Traj_tup = []
    for i in Traj:
        cond = False
        while cond == False:
            rand_tup = (np.random.normal(P_ind_tup[i][0],sig_abs),np.random.normal(P_ind_tup[i][1],sig_abs))
            if rand_tup[0] >= 0 and rand_tup[1] >= 0:
                Traj_tup.append([int(rand_tup[0]+0.5), 
                                 int(rand_tup[1]+0.5)])
                cond = True
    Traj_tup = np.array(Traj_tup)

    shape_new = (np.max(Traj_tup[:,0])+1,np.max(Traj_tup[:,1])+1)

    Pab_r = np.zeros((shape_new[0],shape_new[1]))
    for tup in Traj_tup:
        Pab_r[tup[0],tup[1]] += 1
    Pab_r = Pab_r/sum(sum(Pab_r))

    return Pab_r


# Adding an error on each data point relative to the molecule number with a standard deviation of sig_rel (fraction of the molecule number)
def Sample_Dist_error_rel(P_joint,N,sig_rel):
    
    shape = np.shape(P_joint)
    P_samp = P_joint.reshape((1,shape[0]*shape[1]))[0]

    P_ind = np.arange(len(P_samp))

    P_ind_tup = [[0 for i in range(shape[1])] for i in range(shape[0])]
    for i in range(len(P_ind_tup)):
        for j in range(len(P_ind_tup[0])):
            P_ind_tup[i][j] = (i,j)     
    P_ind_tup = np.array(P_ind_tup).reshape((1,shape[0]*shape[1],2))[0]

    Traj = np.random.choice(P_ind,size = N, p = P_samp)

    Traj_tup = []
    for i in Traj:
        cond = False
        while cond == False:
            rand_tup = (np.random.normal(1,sig_rel),np.random.normal(1,sig_rel))
            if rand_tup[0] >= 0 and rand_tup[1] >= 0:
                Traj_tup.append([int(P_ind_tup[i][0] * rand_tup[0]+0.5), 
                                 int(P_ind_tup[i][1] * rand_tup[1]+0.5)])
                cond = True
    Traj_tup = np.array(Traj_tup)

    shape_new = (np.max(Traj_tup[:,0])+1,np.max(Traj_tup[:,1])+1)

    Pab_r = np.zeros((shape_new[0],shape_new[1]))
    for tup in Traj_tup:
        Pab_r[tup[0],tup[1]] += 1
    Pab_r = Pab_r/sum(sum(Pab_r))

    return Pab_r


# Adding both an absolute error and a relative error to each data point
def Sample_Dist_error_both(P_joint,N,sig_abs,sig_rel):
    
    shape = np.shape(P_joint)
    P_samp = P_joint.reshape((1,shape[0]*shape[1]))[0]

    P_ind = np.arange(len(P_samp))

    P_ind_tup = [[0 for i in range(shape[1])] for i in range(shape[0])]
    for i in range(len(P_ind_tup)):
        for j in range(len(P_ind_tup[0])):
            P_ind_tup[i][j] = (i,j)     
    P_ind_tup = np.array(P_ind_tup).reshape((1,shape[0]*shape[1],2))[0]

    Traj = np.random.choice(P_ind,size = N, p = P_samp)

    Traj_tup = []
    for i in Traj:
        rand_tup_abs = (np.random.normal(P_ind_tup[i][0],sig_abs),np.random.normal(P_ind_tup[i][1],sig_abs))
        rand_tup_rel = (np.random.normal(1,sig_rel),np.random.normal(1,sig_rel))
        if rand_tup_abs[0] >= 0 and rand_tup_abs[1] >= 0 and rand_tup_rel[0] >= 0 and rand_tup_rel[1] >= 0:
            Traj_tup.append([int(rand_tup_abs[0] + P_ind_tup[i][0]*(rand_tup_rel[0]-1) + 0.5), 
                             int(rand_tup_abs[1] + P_ind_tup[i][1]*(rand_tup_rel[1]-1)  + 0.5)])
    Traj_tup = np.array(Traj_tup)

    shape_new = (np.max(Traj_tup[:,0])+1,np.max(Traj_tup[:,1])+1)

    Pab_r = np.zeros((shape_new[0],shape_new[1]))
    for tup in Traj_tup:
        Pab_r[tup[0],tup[1]] += 1
    Pab_r = Pab_r/sum(sum(Pab_r))

    return Pab_r
