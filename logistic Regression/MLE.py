#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 21:21:23 2022

@author: xi
"""


import math
import numpy as np
import matplotlib.pyplot as plt


with open('train.csv',mode='r') as f:
    myList_train=[]
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_train.append(terms)
with open('test.csv',mode='r') as f:
    myList_test = []
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        myList_test.append(terms)
        
def str_to_flo(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data

def add_cons_feature(data):
    label = [row[-1] for row in data]
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 1.0
    for i in range(len(data)):
        temp[i].append(label[i])
    return temp

# convert label {0,1} to {-1,1}
def polar_label(data):
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 2*data[i][-1]-1
    return temp
                   
mylist_train = str_to_flo(myList_train)  #convert to float  types data 
mylist_test = str_to_flo(myList_test) 

train_data = add_cons_feature(polar_label(mylist_train))
test_data = add_cons_feature(polar_label(mylist_test))

train_len = len(train_data)   # NO. of samples
test_len = len(test_data)
dim_s = len(train_data[0]) -1
################## End data processing #############################



def sign_func(x):
    y = 0
    if x> 0:
        y = 1
    else:
        y=-1
    return y

def error_compute(xx,yy):
    cnt = 0
    length =len(xx)
    for i in range(length):
        if xx[i]!= yy[i]:
            cnt = cnt + 1
    return cnt/length

# returns error rate
def predict(wt, data):
    pred_seq =[];
    for i in range(len(data)):
        pred_seq.append(sign_func(np.inner(data[i][0:len(data[0])-1], wt)))
    label = [row[-1] for row in data]
    return error_compute(pred_seq, label) 

def sigmoid(x):
    if x < -100:
        temp = 0
    else:
        temp = 1/(1 + math.e**(-x))
    return temp

def loss_fun(w, data):
    seq = []
#    t1 = 1/(2*var)*np.inner(w,w)
    for row in data:
        temp = -row[-1]*np.inner(w, row[0:dim_s])
        if temp > 100:
            t2 = temp
        else:
            t2 = math.log(1+ math.e**(temp))
        seq.append(t2)
    return sum(seq)

def sgd_grad(w, sample):
     cc = train_len*sample[-1]*(1-sigmoid(sample[-1]*np.inner(w, sample[0:dim_s])))
     return np.asarray( [-cc*sample[i] for i in range(dim_s)])
    
# rate schedule
def gamma(t, gamma_0, d): 
    return gamma_0/(1 + (gamma_0/d)*t)

###### INPUT: w--should be array type, iter_cnt: iteration count

def sgd_single(w, perm, iter_cnt, gamma_0, d):
    w = np.asarray(w)
    loss_seq = []   # sequence of loss fucntion values
    for i in range(train_len):
        w = w - gamma(iter_cnt, gamma_0, d)*sgd_grad(w, train_data[perm[i]])
        loss_seq.append(loss_fun(w, train_data)) 
        iter_cnt = iter_cnt + 1
    return [w, loss_seq, iter_cnt]


# T-- # of epochs
def sgd_epoch(w, T, gamma_0, d):
    iter_cnt = 1
    loss = []
    for i in range(T):
        perm = np.random.permutation(train_len)
        [w, loss_seq, iter_cnt] = sgd_single(w, perm, iter_cnt, gamma_0, d)
        loss.extend(loss_seq)
    return [w, loss, iter_cnt]


w = np.zeros(5)
T = 100      # total no. of epochs
gamma_0 = 2.0
d = 2.0
[wt, loss, tt] = sgd_epoch(w, T, gamma_0, d)
print('train err =', predict(wt, train_data))
print('test err =', predict(wt, test_data))

