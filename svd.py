#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:30:38 2020

@author: jaga
"""

import numpy as np
import timeit
starttime=timeit.default_timer()
'''
     V(T)=Tranpose of V
     Here any m*n matrix A is decomposed as A=UDV(T)
     where U and V are m*m and n*n orthogonal matrix respectively and 
     D is m*n rectangular diagonal matrix.
     and this func svd() returns U, D, V
     '''
def svd(matrix,row,col):
    ''' A(T)=Tranpose of A
        From the theory of SVD V is actually the matrix composed by the eigenvectors
        of A(T)A and U is actually the matrix composed by the eigenvectors of AA(T)
        and D is diag[suqare root of eigenvalues]
        '''
    right=np.dot(np.transpose(matrix),matrix)
    left=np.dot(matrix,np.transpose(matrix))
    right_eig=np.linalg.eigh(right)
    left_eig=np.linalg.eigh(left)
    
    #constructing the diagonal D matrix
    D=np.zeros(min(row,col))
    minimum=min(row,col)
    if minimum==row:
        for i in range(minimum):
            D[i]=np.sqrt(left_eig[0][i])
    else:
        for i in range(minimum):
            D[i]=np.sqrt(right_eig[0][i])
    
    U=(left_eig[1])
    
    V=np.transpose(right_eig[1])
    return (U,D,V)
A=np.array([[0,1,1],[0,1,0],[1,1,0],[0,1,1],[1,0,1]])
U,D,V=svd(A,5,3)

print ("U = \n",U)
print("\n\nSingular Values from the defined svd() \n D = ",D)
print(" \n \n \n V = \n",V)
print("\nTime taken by this code : ",timeit.default_timer()-starttime)

#comparision with np.linalg.svd()
U,D,V=np.linalg.svd(A)
print ("U = \n",U)
print("\n \n D = ",D)
print(" \n \n \n V = \n",V)

# code for calculating the time taken by np.linalg.svd()

starttime=timeit.default_timer()
A=np.array([[0,1,1],[0,1,0],[1,1,0],[0,1,1],[1,0,1]])
U,D,V=np.linalg.svd(A)
print("\nTime taken by np.linalg.svd() : ",timeit.default_timer()-starttime)
