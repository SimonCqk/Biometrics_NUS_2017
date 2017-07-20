#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
:summary: this file includes one functions [list operations]
:author: CHEN QIUKAI
'''


def list_operations():
	L = list()
	L.extend([12, 8, 9])
	print('after add 12,8,9 to the empty list ->', L)
	L.insert(0, 9)
	print('after insert 9 to the head of the list ->', L)
	L = L * 2
	print('after double the list ->', L)
	while 8 in L:
		L.remove(8)
	print('after remove all 8 in the list ->', L)
	print('reverse the list ->', L[::-1])


if __name__ == '__main__':
	list_operations()
