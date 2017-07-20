#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Q1 is to convolve 2 matrix and use python to check the answer
A = 1, 3, 2, 4; 2, 2, 3, 1; 3, 2, 4, 5; 4, 2, 0, 1
B(kernel) = 1, 2, 3; 2, 1, 3; 4, 1, 3
'''
import numpy as np
from scipy import signal

A = np.mat('1, 3, 2, 4; 2, 2, 3, 1; 3, 2, 4, 5; 4, 2, 0, 1')
B = np.mat('1, 2, 3; 2, 1, 3; 4, 1, 3')
ans = signal.convolve(A, B)
print(ans)