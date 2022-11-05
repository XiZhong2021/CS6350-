#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:29:17 2022

@author: xi
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

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
a = np.zeros((n,))
for t in range(T):
    for i in range(D):
        check = train_data[i][n] * np.dot(w, train_data[i][0:n])
        if check <= 0:
            w += r * train_data[i][n] * train_data[i][0:n]
        a = a + w
print("a :", a, "\n")
print("weight vector:\n", w,"\n")     
       
#predict
error = 0
for test in test_data:
    if np.dot(a, test[0:n]) > 0:
        label = 1
    else:
        label = 0
    if label != test[n]:
        error += 1
        
error = error/len(test_data)
print("Average error = ", error)

            
        
    
    
                
