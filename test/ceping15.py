#!/bin/python
# -*- coding: utf8 -*-
import sys
import os
import re

def  miHomeGiftBag(p, M):
    length = len(p)
    flags = [([0]*(M+1)) for i in range(length)]
    for i in range(length):
        flags[i][0] = 1
        
    for i in range(M+1):
        flags[0][i] = 0
        
    flags[0][p[0]] = 1
    for i in range(1, length):
        for j in range(1, M+1):
            if j<p[i]:
                flags[i][j] = flags[i-1][j]
            else:
                flags[i][j] = (flags[i-1][j]) or (flags[i-1][j-p[i]])
                
    return flags[length-1][M]



_p_cnt = 0
_p_cnt = int(raw_input())
_p_i=0
# _p = []
# while _p_i < _p_cnt:
#     _p_item = int(raw_input())
#     _p.append(_p_item)
#     _p_i+=1
s = raw_input()
s = s.split(' ')
_p = [int(a) for a in s]

_M = int(raw_input())

  
res=miHomeGiftBag(_p, _M)

# print str(int(res)) + "\n"
print int(res)