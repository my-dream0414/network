# -*- codeing = utf-8 -*-
# @Time : 2024/9/9 19:46
# @Author : Luo_CW
# @File : 感知机.py
# @Software : PyCharm

def AND(x1, x2):
	w1, w2, theta = 0.5, 0.5, 0.7
	tmp = x1*w1 + x2*w2
	if tmp <= theta:
		return 0
	elif tmp > theta:
		return 1

if __name__ == '__main__':
    x1 = 2
    x2 = 3
    d = AND(2,3)
    print(d)