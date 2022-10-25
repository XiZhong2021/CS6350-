import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


#Read train_data as a array
train_data = []
with open ('train.csv','r') as f:
    i =0
    for line in f:
        terms = line.strip().split(',')
        terms = [float(x) for x in terms]
        terms = np.array(terms)
        train_data.append(terms)

train_data = np.array(train_data)
# print(train_data)

T =10000
iteration = [i for i in range(T)]


r = 0.000001

w = np.zeros((7,))

Cost = []
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

print("Iteration =", T,"\n")
print("Learned weight vector:\n", w,"\n")
print("Learning rate =" , r,"\n")
print("cost funtion value = ", Cost[T-1])
# print("Cost function value vector:\n", Cost,"\n")

plt.scatter(iteration, Cost)
plt.show()
