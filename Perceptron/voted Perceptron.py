#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:47:14 2022

@author: xi
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as geek  
data = []

#Read train as a array
train = []
with open ('train.csv','r') as f:
    i =0
    for line in f:
        terms = line.strip().split(',')
        terms = [float(x) for x in terms]
        terms = np.array(terms)
        train.append(terms)
train = np.array(train)
data.append(train)

#Read test as a array
test = []
with open ('test.csv','r') as f:
    i =0
    for line in f:
        terms = line.strip().split(',')
        terms = [float(x) for x in terms]
        terms = np.array(terms)
        test.append(terms)
test = np.array(test)
data.append(test)
# print(test)


r = 0.1
T = 11

train_data = data[0]
n = len(train_data[0])-1
# print(n)
D = len(train_data)

test_data = data[1]
w = np.zeros((n,))
m = 0
C = [0]
w_list = [w]
for t in range(T):
    for i in range(D):
        check = train_data[i][n] * np.dot(w, train_data[i][0:n])
        if check > 0 :
            C[m] += 1
            continue
        else:
            w += r * train_data[i][n] * train_data[i][0:n]
            m += 1
            C.append(0)
            w_list.append(w)
            
print("learned weight vectors: \n", w)  
# print("Number of updates =", len(C))          
print("Number of correct prediction =", sum(C))
    
error = 0
for test in test_data:
    pre = 0
    for i in range(len(C)):
        pre += C[i] * geek.sign(np.dot(w, test[0:n]))
    if geek.sign(pre) < 0:
        label = 0
    else:
        label = 1
    if label != test[n]:
        error += 1

print("Error is =", error/len(test_data))



                
