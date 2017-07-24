#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from numpy import linalg

# (c)
A = np.mat('100,100;100,100.01')
b = np.mat('2;2.0001')
x = linalg.solve(A, b)
print(x)
'''
 when the b is recorrected, x turn out to be [0.01;0.01],
 which is the right answer. And the error in A was squared by x, then shown in b.
'''

# (d)
det_A = linalg.det(A)
con = max(linalg.svd(A)[1]) / min(linalg.svd(A)[1])
print('Determinant={0} Condition number={1}'.format(det_A, con))
'''
explain:
Since the kp(A) is so big ,so when there is error in b,
the corresponding error in x is multiplied many times, and it seems quietly different.

'''
