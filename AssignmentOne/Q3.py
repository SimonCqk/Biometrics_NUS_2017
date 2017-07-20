#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
:summary: this file includes one function [str spilt/join]
:author: CHEN QIUKAI
'''


def change_delimiter(string: str):
	return '-'.join(string.split())


if __name__ == '__main__':
	string = input('Input a string for test.\n')
	print(change_delimiter(string))
