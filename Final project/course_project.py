#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:35:43 2022

@author: xi
"""

import numpy as np
import math
import random
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Choose features
# =============================================================================
train_data = []
with open ('train.csv','r') as f:
    g = 0
    for line in f:
        terms = line.strip().split(',')
        terms = list(terms)
        if g >0:
            for i in range(13):
                if terms[i] == 'male':
                    terms[i] = '1'
                if terms[i] == 'female':
                    terms[i] = '0'
                if terms[i] == "":
                    terms[i] = '10000'  #missing feature is labeled by 10000
            train_data.append(terms)
        g = g+1      
train_data = np.array(train_data)

train = []
useful_feature = np.array([2, 5, 6, 7, 8, 10, 1])
for i in range(len(train_data)):
    example = []
    for j in useful_feature:
        example.append(train_data[i][j])
    example = np.array(example, dtype='f')
    train.append(example)
train = np.array(train)

    #missing data is column 2-age
    # =============================================================================
    # #replace missing data and label
    # =============================================================================
missing = list(np.where(train == 10000)[0])
#average in label
negtive = 0
neg_count = 0
positive = 0
posi_count = 0


#list age and passenger
for i in range(len(train)):
    if train[i][6] == 0:
        train[i][6] = -1
        if i not in missing:
            neg_count += 1
            negtive += train[i][2]
            
    if train[i][6] == 1:
        if i not in missing:
            posi_count += 1
            positive += train[i][2]
            
    # =============================================================================
    #  #Avergae
    # =============================================================================
posi_averge = positive/posi_count
neg_average = negtive/neg_count
print(posi_averge,neg_average)

for i in missing:
    if train[i][6] == -1:
        train[i][2] = neg_average
    else:
        train[i][2] = posi_averge



test_data = []
with open ('test.csv','r') as f:
    g = 0
    for line in f:
        terms = line.strip().split(',')
        terms = list(terms)
        if g >0:
            for i in range(12):
                if terms[i] == "":
                    terms[i] = '10000' #missing feature is labeled by 10000
                if terms[i] == 'male':
                    terms[i] = '1'
                if terms[i] == 'female':
                    terms[i] = '0'
            test_data.append(terms)
        g = g+1
test_data = np.array(test_data)

test = []
useful_feature = np.array([1, 4, 5, 6, 7,9])
for i in range(len(test_data)):
    example = []
    for j in useful_feature:
        example.append(test_data[i][j])
    example = np.array(example, dtype='f')
    test.append(example)
test = np.array(test)

    #missing data is column 2-age, 5-fare
    # =============================================================================
    # #replace missing data
    # =============================================================================
missing = np.where(test == 10000)
missing_age = missing[0][np.where(missing[1] == 2)[0]]
missing_fare = 152
#average in example
age = 0
fare = 0
for i in range(len(test)):
    if i not in missing_age:
        age +=  test[i][2]
    if i != 152:
        fare += test[i][5]
age_ave = age/(len(test)-len(missing_age))
fare_ave = fare/(len(test)-1)

for i in missing_age:
    test[i][2] = age_ave
test[152][5] = fare_ave
# print("test data : \n", test)
# print("len_test is ", len(test),len(test[0]))


# =============================================================================
# #SDG
# =============================================================================

# =============================================================================
# #Learning
# =============================================================================
w = np.zeros((7,))

T = 400

m = len(train)

shuffle = [i for i in range(m)]

C = 0.5

gamma = 0.000001
for t in range(T):
    #shuffle data
    random.shuffle(shuffle)
    for i in range(m):
        x = list(train[i][0:6])
        x.append(1)
        x = np.array(x)
        
        y = train[i][6]
        #compute gradient
        gradient = []
        for j in range(7):
            gra = m * math.e**(-y*np.dot(w,x)) * (-y*x[j]) / (1+math.e**(-y*np.dot(w,x))) + 2*C*w[j]
            # print(gra)
            gradient.append(gra)
        gradient = np.array(gradient)
        w = w - gamma * gradient
        # print(gradient)
print(w)

# =============================================================================
# #Predict
# =============================================================================
n = len(test)
Predict = []
Possibility = []
for i in range(n):
    x = list(test[i])
    x.append(1)
    x = np.array(x)
    
    posibility = 1/(1 + math.e**(-np.dot(w,x)))
    
    Possibility.append(posibility)
    
    if posibility > 1/2:
        Predict.append(1)
    else:
        Predict.append(0)
        
Predict = np.array(Predict)

# print(Possibility)
# print(Predict)

print(len(np.where(Predict == 1)[0]))
print(np.max(Possibility), np.min(Possibility))

# =============================================================================
# #CSV file
# =============================================================================

pd.DataFrame(Predict).to_csv('Prediction.csv')




# # =============================================================================
# # #Define label
# # =============================================================================
# #age
# Data = [train,test]
# for d in range(2):
#     data = Data[d]
#     for i in data[:,2]:
#         if i>=0 and i<16:
#             i=0
#         elif i>=16 and i<32:
#             i=1
#         elif i>=32 and i<48:
#             i=2
#         else:
#             i = 4
#     if d == 0:
#         train = data
#         print(data)   #-------------------

    


# #fare
# for i in train[:,5]:
#     if i>=0 and i<50:
#         i = 0
#     else: 
#         i =1

# # =============================================================================
# # #define dictionary
# # =============================================================================
# at = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare']

# Value = {}
# Value['pclass'] = [1,2,3]
# Value['sex'] = [0,1]
# Value['age'] = [0,1,2,3,4]
# Value['sibsp'] = [i for i in range(9)]
# Value['parch'] = [i for i in range(8)]
# Value['fare'] = [0,1]

# Label = [0,1]

# # =============================================================================
# # Define funtions
# # =============================================================================
# def entro(p):  #p is the probability vector
#     H = 0
#     for i in range(len(p)):
#         if p[i] >0:
#             H += -(p[i])*math.log(p[i],2)
#     return H

# def gain(a,x,y):
#     return a-x@y

# Deep = 1
        

