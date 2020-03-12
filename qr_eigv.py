#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 03:03:18 2020

@author: jaga
"""

import numpy as np
a=np.array([[5,-2],[-2,8]])
Q,R=np.linalg.qr(a)
print ("QR decomposition of:",a, "is\n", "Q=",Q, "\n R=",R )
print(np.allclose(a,np.dot(Q,R)))

'''diag_check() function returns 'false' when any of non diagonal elements is greater
  than 0.0001 and returns 'true' when all the non diagonals are less than this '''

def diag_check(matrix,order):
       count=0  # To track whether control enters into second loop or not 
       for i in range(order):
           for j in range(order):
               if i!=j and abs(matrix[i][j])>0.0001:
                   count=1
                   return 'false'  # i.e if one of the off-diagonal element is less 
                   break           # than 0.0001 it's obviously non-diag matrix
           break
       if count==0:
           return 'true'
while diag_check(a,2)=='false':
     Q,R=np.linalg.qr(a)   #  Eigenvalues calculation using QR-Decomposition
     a=np.linalg.multi_dot([np.transpose(Q),a,Q])
eigvalues=[]  # a list to store the diagonal elements i.e the eigenvalues
for i in range(len(a)):
    eigvalues.append(a[i][i])
print("Using QR Eigenvalues: ",eigvalues)
print("Using np.linalg.eigh():  ",np.linalg.eigh(a)[0])
