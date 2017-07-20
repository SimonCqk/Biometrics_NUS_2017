#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
:summary: this file include the functions which solve the cryptarithmetic puzzle [PZCZ * 15 = MUCHZ]
:author: CHEN QIUKAI
'''


def isPZCZ(n) -> bool:
	'''
	:param n: Input number, assumed to be in format PZCZ
	:return: Returns True if n has the format PZCZ, False otherwise.
	'''
	if str(n)[1] == str(n)[-1]:
		return True
	else:
		return False


def isProductMUCHZ(pzcz: int, product: int) -> bool:
	'''
	:param pzcz: Input number, assumed to be in format PZCZ
	:param product: Input number
	:return: Returns True if product passes all 3 checks, False otherwise.
	'''
	pzcz_str, product_str = str(pzcz), str(product)
	if len(product_str) == 5 and product_str[-1] == pzcz_str[-1] and product_str[2] == pzcz_str[2]:
		return True
	else:
		return False


if __name__ == '__main__':
	S = [(x, x * 15) for x in range(1000, 10000) if isPZCZ(x) and isProductMUCHZ(x, x * 15)]
	print(S)
