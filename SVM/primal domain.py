#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:37:02 2022

@author: xi
"""

import numpy as np
import random

data = []

train = []
with open ('train.csv','r') as f:
    i =0
    for line in f:
        terms = line.strip().split(',')
        terms = [float(x) for x in terms]
        terms = np.array(terms)
        if terms[-1] == 0:
            terms[-1] = -1
        train.append(terms)
train = np.array(train)
data.append(train)
# print(train)

#Read test as a array
test = []
with open ('test.csv','r') as f:
    i =0
    for line in f:
        terms = line.strip().split(',')
        terms = [float(x) for x in terms]
        terms = np.array(terms)
        if terms[-1] == 0:
            terms[-1] = -1
        test.append(terms)
test = np.array(test)
data.append(test)
# print(test)


gamma = 0.01

C_list = [100/873, 500/873, 700/873]

T = 100

gamma = 0.001

a = 10

for C in C_list:
    print("---------------------")
    print("C=", C,"\n")
        
    for d in range(2):
        train_data = data[d]
        N = len(train_data)
        example_list = [i for i in range(N)]
                
        w_long = np.zeros((5,))
        w_short = np.zeros((4,))
        
        for t in range(T):
            random.shuffle(example_list)
            
            # rate = gamma/(1+t*(gamma/a))
            rate = gamma/(1+t)
    
            for i in example_list:
                x_long = list(train_data[i][0:4])
                x_long.append(1)
                x_long = np.array(x_long)
                
                w_0 = list(w_short)
                w_0.append(0)
                w_0 = np.array(w_0)
                
                check = train_data[i][4] * np.dot(w_long,x_long)
                if check <=1:
                    w_long = w_long - rate * w_0  + rate * C * N * train_data[i][4] * x_long
                else:
                    w_short = (1-rate)*w_short
        
        #predict
        error = 0
        for test in train_data:
            if np.dot(w_long[0:4], test[0:4]) > 0:
                label = 1
            else:
                label = -1
            if label != test[4]:
                error += 1
                
        error = error/len(train_data)
        if d == 0:
            print("Error of training data is", error)
            print("model parameters learned of training data is \n", w_long,"\n")

        if d == 1:
            print("Error of test data is ", error)
            print("model parameters learned of test data is \n", w_long,"\n")
    
            
    