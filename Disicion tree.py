#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:55:32 2022

@author: xi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 22:42:29 2022

@author: xi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:35:02 2022

@author: xi
"""

import numpy as np
import math

#Read train_data as a array
train_data = []
with open ('train.csv','r') as f:
    i =0
    for line in f:
        terms = line.strip().split(',')
        terms = np.array(terms)
        train_data.append(terms)

train_data = np.array(train_data)
#print(train_data)

# =============================================================================
# #define dictionary
# =============================================================================
attribute = ['buying', 'maint', 'doors', 'persons','lug_boot', 'safety']

Value = {}
Value['buying'] = ['vhigh', 'high','med','low']
Value['maint'] = ['vhigh', 'high','med','low']
Value['doors'] = ['2','3','4','5more']
Value['persons'] = ['2','4','more']
Value['lug_boot'] = ['small','med','big']
Value['safety'] = ['low','med','high']

Label = ['unacc','acc','good','vgood']

# =============================================================================
# Define funtions
# =============================================================================
def entro(p):  #p is the probability vector
    H = 0
    for i in range(len(p)):
        if p[i] >0:
            H += -(p[i])*math.log(p[i],2)
    return H
  
def ME(n): #n is the number vector
    if np.max(n) >0:
        major = np.max(n)
        normal = np.sum(n)
        ME = (normal-major)/normal
        return ME
    else:
        return 0

def GI(p): #p is the probability vector
    GI = 1
    for i in range(len(p)):
        GI -= p[i]**2
    return GI

def gain(a,x,y):
    return a-x@y

Deep = 2


variant = 0
#H=0,ME=1,GI=2

dataset = {}  #data[deep][attribute][value]
dataset[0] = {'data': [train_data],
              'attri_path': [[]],
              'value_path': [[]]}


#Store all paths
Path = {'data':[],
        'attri_path': [],
        'value_path': [],
        'label':[]
        }


# =============================================================================
# #first partition
# =============================================================================
count_leaf = 0     #number of leave in this deep

for deep in range(Deep):
    # print("\n For deep=",deep)
    dataset[deep+1] = {'data':[],
                       'attri_path':[],
                       'value_path':[]} #store next subset
    
    count_subset = 0  #number of subset in this deep
    
    for index in range(len(dataset[deep]['data'])):
        data = dataset[deep]['data'][index]
        attri_path = dataset[deep]['attri_path'][index]
        value_path = dataset[deep]['value_path'][index]
        size = len(data)
        
        # print("in deep = ", deep, ",", index, "-th data, the used feature:",attri_path,"\n")
        
        #number of each label
        # attribute_remaining = [ele for ele in attribute if ele not in attri_path]
        n = np.zeros((len(Label),)).astype(int) #number vector
        
        for i in range(len(Label)):
            n[i] = len(np.where(data[:,6] == Label[i])[0])  
        
        #entropy
        Hv = entro(n/np.sum(n))
        
        #ME
        MEv= ME(n)
        
        
        #GI value
        GIv= GI(n/np.sum(n))
        
        # if deep ==3:
        #     if index == 4:
        #         # print(n)
        #         # print(data)
        #         print("Total entropy = ", Hv)
        #         print("Total ME =",MEv)
        #         print("Total GI=",GIv,"\n")

        # # =============================================================================
        # # #For each attribute, compute the gain
        # # =============================================================================
        H_Gain = []
        ME_Gain = []
        GI_Gain = [] #stores gains of each attribute
        
        #update remaining attribute
        new_attribute = []
        for feature in attribute:
            if feature not in dataset[deep]['attri_path'][index]:
                new_attribute.append(feature)
        # print("in deep = ", deep, ",", index,"-th data")
        # print("used attributes",dataset[deep]['attri_path'][index])
        # print("the remaining feature:",new_attribute,"\n")
        
        for feature in new_attribute:
            # if deep ==0:
            #     if index ==0:
            #         print("attribute =",feature)
        
            position_feature = attribute.index(feature)
            # print(position_feature)
            
            n_value = [] #store the number of each value
            p_value = []
            H_value = []
            ME_value = []
            GI_value = []
            for v in range(len(Value[feature])):  
                # print("value = ", value)
                
                value_where = np.where(data[:,position_feature] == Value[feature][v])[0]   #row index of this value
                n_value.append(len(value_where))  #number of this value
                
                if len(value_where) >0:
        
                    value_data = data[value_where]  #data with this value
                    
                    # =============================================================================
                    # for this value, partition the data by label, compute the entropy
                    # =============================================================================
                    n_label = []
                
                    for label in Label:
                        n_label.append(len(np.where(value_data[:,6] == label)[0])) #for this label, the number of examples
                
                    p_label = np.array(n_label)/len(value_where)  #probablity of all labels in this value
                    # print("probability vector of label:\n", p_label,"\n")
                    
                    #compute the entropy for a value
                    H_value.append(entro(p_label))
                    
                    #ME of each value
                    ME_value.append(ME(n_label))
                    
                    #GI of each value
                    GI_value.append(GI(p_label))
                    
                else:
                     H_value.append(0)
                     ME_value.append(0)
                     GI_value.append(0)
                
            p_value = np.array(n_value)/size

            # =============================================================================
            #  Combine above entropy to compute gain
            # =============================================================================
            
            information_gain = gain(Hv,p_value,H_value)
            H_Gain.append(information_gain)
            
            ME_gain = gain(MEv, p_value,ME_value)
            ME_Gain.append(ME_gain)
        
            GI_gain = gain(GIv, p_value,GI_value)
            GI_Gain.append(GI_gain)
            

        # =============================================================================
        # #determine the attribute
        # =============================================================================
        # print("H_Gain of all feature:\n", H_Gain,"\n")
        # print("ME_Gain of all feature:\n", ME_Gain,"\n")
        # print("GI_Gain of all feature:\n", GI_Gain,"\n")
        
        if variant == 'H':
            Gain = H_Gain
        elif variant == 'ME':
            Gain = ME_Gain
        else:
            Gain = GI_Gain
            
        attribute_name = new_attribute[Gain.index(max(Gain))]
        attribute_number = attribute.index(attribute_name)
        # if deep == 1:
        #     if index == 1:
        #         print("For ",deep,index, "subset")
        #         print("chosen attribute =", attribute_name,"\n")
                # print("chosen attribute column = ", attribute_number,"\n")

        # =============================================================================
        # partition to subsets
        # =============================================================================
        for value in Value[attribute_name]:
            subset = data[np.where(data[:,attribute_number] == value)[0]]
            
            if len(subset) > 0:
                number_case = len(subset)
                # print("For value = ", value, "# of case = ",number_case)
                        
                attri_copy = attri_path.copy()
                attri_copy.append(attribute_name)
                
                value_copy = value_path.copy()
                value_copy.append(value)
                
                #check leaf
                label_check = []
                for example in range(len(subset)):
                    if subset[example][6] not in label_check:
                        label_check.append(subset[example][6])
                                            
                        
                if len(label_check) == 1: #leaf
                    Path['data'].append(subset)
                    Path['attri_path'].append(attri_copy)
                    Path['value_path'].append(value_copy)
                    Path['label'].append(label_check[0])
                    
                    # print("=========Acheive leaf(Label) = ",label_check)
                    # print("Obtained path:", Path['attri_path'][count_leaf])
                    # print("Obtained leaf:", Path['value_path'][count_leaf],"\n")
                    # print("leaves", number_case,"\n")
                    
                    count_leaf += 1
                    
                elif len(label_check) >1:
                    dataset[deep+1]['data'].append(subset)
                    dataset[deep+1]['attri_path'].append(attri_copy)
                    dataset[deep+1]['value_path'].append(value_copy)
                    
                    # print(number_case ,"non-leaves","\n")
                    # print(dataset[deep+1]['data'][count_subset])
                    # print(dataset[deep+1]['attri_path'][count_subset],)
                    # print(dataset[deep+1]['value_path'][count_subset],"\n")
                    
                    
                    count_subset += 1
                    
#give label
for i in range(len(dataset[Deep]['data'])):
    subset = dataset[Deep]['data'][i]
    number_label =[]
    for i in range(len(Label)):
        number_label.append(len(np.where(subset[:,6] == Label[i])[0]))
    # print(number_label)
    label = Label[number_label.index(max(number_label))]
    
    Path['data'].append(subset)
    Path['attri_path'].append(dataset[Deep]['attri_path'][i])
    Path['value_path'].append(dataset[Deep]['value_path'][i])
    Path['label'].append(label)    

for i in range(len(Path['data'])):
    print(Path['attri_path'][i])
    print(Path['value_path'][i])
    print(Path['label'][i],"\n")
    
# =============================================================================
# #check correct
# =============================================================================
# number_leaves = 0
# for i in range(len(Path['data'])):
#     number_leaves += len(Path['data'][i])
#     # print(Path['attri_path'][i])
#     # print(Path['value_path'][i])
#     # print(Path['label'][i],"\n")

# print("--------------------------------")   
# print("Leaves number =",  number_leaves,"\n" )


# # print("Non-leaf:\n")
# number_noleaves = 0
# # print(len(dataset[Deep]['data']))
# for i in range(len(dataset[Deep]['data'])):
#     number_noleaves += len(dataset[Deep]['data'][i])
    
# print("Non-leaves number =",  number_noleaves )
    
    





