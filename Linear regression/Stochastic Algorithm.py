import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

data = []

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
# print(train_data)

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


T =1000
iteration = [i for i in range(T)]
print("Iteration =", T,"\n")
print('---------------------------------')

r = 0.000001

w = np.zeros((7,))
for d in range(2):
    Cost = []
    train_data = data[d]
    for t in range(T):
        for i in range(len(train_data)):
            #update w
            for j in range(7):
                w[j] += r*(train_data[i][7]-np.dot(w, train_data[i][0:7]))*train_data[i][j]    
    
        cost = 0
        for j in range(len(train_data)):
            cost += (train_data[j][7] - np.dot(w,train_data[j][0:7]))**2
        cost = cost/2
        Cost.append(cost)
        
    if d == 0:
        print("dataset = train data", "\n")
    if d == 1:
        print("dataset = test data"," \n")
        
    print("Learned weight vector:\n", w,"\n")
    print("Learning rate =" , r,"\n")
    print("cost funtion value = ", Cost[T-1])
    # print("Cost function value vector:\n", Cost,"\n")
    print("------------------------------------------")
    plt.scatter(iteration, Cost)
    plt.show()
