#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
:summary: this file includes one function [some str operations]
:author: CHEN QIUKAI
'''


def str_operations(string: str):
	print('length of this string:%s' % len(string))
	print('convert to lower case:%s' % string.lower())
	print('convert to upper case:%s' % string.upper())
	print('and reverse it:%s' % string[::-1])


if __name__ == '__main__':
	string = input('Input a string for test.\n')
	str_operations(string)
