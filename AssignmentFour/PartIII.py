#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

P = (2, 3)
C = (3, 2)
THETA = [i * np.pi / 4 for i in range(8)]
R = [np.mat('{cos},-{sin},0;{sin},{cos},0;0,0,1'.format(cos=np.cos(t), sin=np.sin(t))) for t in THETA]
T1 = np.mat('1,0,{0};0,1,{1};0,0,1'.format(-C[0], -C[1]))
T2 = np.mat('1,0,{0};0,1,{1};0,0,1'.format(C[0], C[1]))

p_list = []
for i in range(len(THETA)):
	pv = T2 * R[i] * T1 * np.matrix([[P[0], P[1], 1]]).T
	p_list.append(pv[0:2] / pv[2])

plt.plot(P[0], P[1], 'b.')
for i in p_list:
	plt.plot(i[0], i[1], 'b.')

plt.annotate('Origin P (2,3)', P, xytext=(2.2, 3.2), arrowprops=dict(arrowstyle='->'))
plt.show()
