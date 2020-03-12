#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 03:14:35 2020

@author: jaga
"""

import numpy as np
A=np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
x0=np.array([1,1,0]) # initial guess
print("Actual Dominant eigenvalue : ",max(np.linalg.eigh(A)[0]))
'''  
   For a given accuarcy (here 0.01) it might be seemed that we need actual eigen value
   to terminate the iteration.But I think that's not the power of 'power method'.
   
   In this code dominant eigenvalue and eigenvector is calculated by two different algorithim 

 '''
 # Applying the formula \eig(cal)-eig(actl)\<=sqrt((A**(m+1)x.A**(m+1)x)/(A**(m)x.A**(m)x)-1)
 # which can be derived to get the bound on error
m=1
x=np.dot(np.linalg.matrix_power(A,m),x0)
Ax=np.dot(A,x)
domin_eigvlu=np.dot(Ax,x)/np.dot(x,x)
domin_eigvtr=x/np.linalg.norm(x)
while np.sqrt(np.dot(Ax,Ax)/(np.dot(x,x)*domin_eigvlu**2)-1)>=0.01:
    m=m+1
    x=np.dot(np.linalg.matrix_power(A,m),x0)
    Ax=np.dot(A,x)
    domin_eigvlu=np.dot(Ax,x)/np.dot(x,x)
    domin_eigvtr=x/np.linalg.norm(x)
print("\nDominant eigenvalue using Algo-1 :",domin_eigvlu)
print("Dominant eigenvector using Algo-1 :",domin_eigvtr)


# Using the algo that when the difference between the eigenvalue at mth and 
# (m+1)th step is less than 0.01 then the iteration will ba stopped
m=1
x=np.dot(np.linalg.matrix_power(A,m),x0)
Ax=np.dot(A,x)
domin_eigvlu_mth=np.dot(Ax,x)/np.dot(x,x)
domin_eigvtr=x/np.linalg.norm(x)
domin_eigvlu_next_mth=0.1 # initial value assigned to (m+1)th for entering into the loop
while abs(domin_eigvlu_mth-domin_eigvlu_next_mth)>=0.01:
    m=m+1
    domin_eigvlu_mth=domin_eigvlu_next_mth  #  assignment of (m+1)th eigenvalue to mth 
    x=np.dot(np.linalg.matrix_power(A,m),x0)
    Ax=np.dot(A,x)
    domin_eigvlu_next_mth=np.dot(Ax,x)/np.dot(x,x) # (m+1)th getting updated by the formula
    domin_eigvtr=x/np.linalg.norm(x)
print("\nDominant eigenvalue using Algo-2 :",domin_eigvlu_mth)
print("Dominant eigenvector using Algo-2 :",domin_eigvtr)
