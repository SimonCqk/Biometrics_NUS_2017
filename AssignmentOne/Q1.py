#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
:summary: this file includes two functions [sum from 100 to 200]
:author: CHEN QIUKAI
'''


def while_version():
	ans, counter = 0, 100
	while counter <= 200:
		ans += counter
		counter += 1
	return ans


def for_version():
	ans = 0
	for cnt in range(100, 201):
		ans += cnt
	return ans


if __name__ == '__main__':
	print('The sum is [for] :', while_version())
	print('The sum is [while] :', for_version())
