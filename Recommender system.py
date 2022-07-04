#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def matrix_factorisation(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.01):

    Q = Q.T

    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # Calculating error
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # Calculating gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

        eR = np.dot(P,Q)

        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
      
    # 0.001: local minimum
        if e < 0.001:

            break

    return P, Q.T


# In[3]:


R = [

     [5,3,2,3,1],

     [1,0,2,4,3],

     [1,4,0,1,0],

     [0,5,2,0,2],

     [1,0,1,3,0],
    
    ]

R = np.array(R)
N = len(R)
M = len(R[0])
K = 3
 
P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorisation(R, P, Q, K)

nR = np.dot(nP, nQ.T)

nR


# In[ ]:




