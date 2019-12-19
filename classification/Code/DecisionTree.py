#!/usr/bin/env python
# coding: utf-8

# In[131]:


import pandas as pd
import numpy as np
import math
import sys
from statistics import mean
from random import randrange
from random import seed
from sklearn import preprocessing
import random


# In[132]:


# Data structure for saving tree where each node contains values: split value and index of split value
class Node:
    def __init__(self, data, index):

        self.left = None
        self.right = None
        self.data = data
        self.index = index
    def insert_left (self,data, index):
        self.left=Node(data,index)
    def insert_right(self,data,index):
        self.right = Node(data,index)
    def PrintTree(self):
        if self.left:
            self.left.PrintTree()
        print( self.data, self.index),
        if self.right:
            self.right.PrintTree()
    def display(self):
        lines, _, _, _ = self._display_aux()
        for line in lines:
            print(line)
    def _display_aux(self):
        #Returns list of strings, width, height, and horizontal coordinate of the root.
        # No child.
        if self.right is None and self.left is None:
            if self.index in dict_tree.keys():
                val = dict_tree[self.index]
                dat = val[self.data]
                #print(dat)
                line = '%s' % dat+"("+str(self.index)+")"
            else:
                dat = round(self.data,2)
                line = '%s' % dat
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            if self.index in dict_tree.keys():
                val = dict_tree[self.index]
                dat = val[self.data]
                #print(dat)
                s = '%s' % dat+"("+str(self.index)+")"
            else:
                dat = round(self.data,2)
                s = '%s' % dat
                #s = '%s' % self.data#+"("+str(self.index)+")"
            #s = '%s' % self.data#+"   "+str(self.index)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            if self.index in dict_tree.keys():
                val = dict_tree[self.index]
                dat = val[self.data]
                #print(dat)
                s = '%s' % dat+"("+str(self.index)+")"
            else:
                dat = round(self.data,2)
                s = '%s' % dat
                #s = '%s' % self.data#+"("+str(self.index)+")"
            #s = '%s' % self.data#+"   "+str(self.index)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        if self.index in dict_tree.keys():
            val = dict_tree[self.index]
            dat = val[self.data]
            #print(dat)
            s = '%s' % dat+"("+str(self.index)+")"
        else:
            dat = round(self.data,2)
            s = '%s' % dat
            #s = '%s' % self.data#+"("+str(self.index)+")"
        #s = '%s' % self.data#+"   "+str(self.index)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2


# In[133]:


# User Input for Dataset & Number of Folds
file_name = input("Enter file name: ")
if(file_name=="project3_dataset4.txt"):
    choice=1
else:
    choice=0
if(choice==0):
    fold = int(input("Enter number of folds: "))


# In[134]:


# Reading file into Dataframe
Data = pd.read_csv(file_name, sep='\t', lineterminator='\n', header=None)


# In[135]:


#Data_ori = Data.copy()


# In[136]:


Data


# In[137]:


# Function to check if the dataset has nominal(string) and/or continuous(numerical) and return the columns containing strings
def check_string(data):
    index=[]
    flag = False
    #print(data.iloc[0])
    for i in range(len(data.iloc[0])):
        #print(data[i][0])
        st = str(data[i][0])
        if(st.replace('.','',1).isdigit() == False):
            flag=True
            index.append(i)
    return flag, index      


# In[138]:


# Funtion to convert string to its equivalent one hot enconding value
def one_hot_encoding(data):
    le = preprocessing.LabelEncoder()
    unique_vals = np.unique(data)
    #print(unique_vals)
    le.fit(unique_vals)
    #print(data.shape)
    data = le.transform(data)
    #print(data)
    return data


# In[139]:


# Check if the dataset has any string values and converting the string values to its one hot encoding values
dict_tree={}
flag, str_index = check_string(Data)
#print(flag, str_index)
np_data = Data.to_numpy()
original = np.array(np_data)
for i in str_index:
    #print(i)
    np_data[:,i]=one_hot_encoding(np_data[:,i])
    a=np_data[:,i]
    b=original[:,i]
    tree={}
    for d in range(len(a)):
        #print(a[d],":",b[d])
        tree[a[d]]=b[d]
    #print(tree)
    dict_tree[i]=tree
Data=pd.DataFrame(np_data)
#Data


# In[140]:


print(dict_tree)


# In[141]:


# Function to calculate GINI Index
def gini_index(left,right,label):
    total_length = len(left)+len(right)
    #print(type(left), np.shape(left))
    if(len(left)!=0):
        left_label = np.asarray(left)[:,-1]
    if(len(right)!=0):
        right_label = np.asarray(right)[:,-1]
    #print("total length: ",total_length)
    gini = float(0)
    gl_sum = float(0)
    gr_sum = float(0)
    for l in label:
        if(len(left)!=0):
            gl=float(0)
            gl = np.count_nonzero(left_label == l)/len(left_label)
            gl_sq = gl*gl
            gl_sum +=gl_sq
        if(len(right)!=0):
            gr=float(0)
            gr = np.count_nonzero(right_label == l)/len(right_label)
            gr_sq = gr*gr
            gr_sum += gr_sq
    if(len(left)==0):
        gini = (1-gr_sum)*(len(right_label)/total_length)
    elif(len(right)==0):
        gini = (1-gl_sum)*(len(left_label)/total_length)
    else:
        gini = ((1-gl_sum)*(len(left_label)/total_length))+((1-gr_sum)*(len(right_label)/total_length))
    return gini


# In[142]:


# Function to split tree depending on split value into left and right subsets
def getTreeSplit(data,val,index):
    left=[]
    right=[]
    for d in data:
        #print(type(d), d.shape)
        if d[index]<val:
            left.append(d)
        else:
            right.append(d)
    return left,right


# In[143]:


# Function to get the minimum GINI Index value, its index, left and right subsets
def Tree(data):
    #print("data to create Tree: ",data.shape)
    unique_label = np.unique(data[:,-1])
    col = len(data[0])-1
    rows = len(data)
    min_gini = sys.float_info.max
    index = 0
    left_tree=[]
    right_tree=[]
    value = 0
    for c in range(col):
        for r in range(rows):
            #print(data[r][c])
            left,right=getTreeSplit(data,data[r][c],c)
            #print(np.asarray(left).shape, np.asarray(right).shape)
            gini = gini_index(left,right,unique_label)
            #print("Index: ", c, gini, data[r][c])
            if gini<min_gini:
                min_gini = gini
                index = c
                left_tree=left
                right_tree=right
                value = data[r][c]
    #Remove already selected feature from being considered in future split decisions
    if len(left_tree)!=0:
        np.delete(left_tree, index, 1)
    if len(right_tree)!=0:
        np.delete(right_tree, index, 1)
    print("Gini: ",gini)    
    return  index, value, left_tree, right_tree


# In[152]:


# Function to create Tree based on root, left and right subsets.
def createTree(root,left,right):
    #print(type(left))
    #print("left: ",len(left), "right: ",len(right))
    if((len(left)==0) or (len(right)==0)):
        if(len(left)==0):
            r_label = np.asarray(right)[:,-1]
            r_label=r_label.tolist()
            max_val = max(r_label,key=r_label.count)
        else:
            l_label = np.asarray(left)[:,-1]
            l_label=l_label.tolist()
            max_val = max(l_label,key=l_label.count)
        root.insert_left(max_val,None)
        root.insert_right(max_val,None)
        return
    else:
        if(len(left)==1):
            #print("$$$$$$$Left 0: ",left[0][-1] )
            root.insert_left(left[0][-1],None)
        elif(len(left)>1):
            #Check if class is same in remaining subset
            label_set = np.unique(np.asarray(left)[:,-1])
            if(len(label_set>1)):
                node_index, node_val, left_node, right_node=Tree(np.asarray(left))
                print("Left: ",len(left_node))
                print("Right: ",len(right_node))
                #print("Index: ", node_index, node_val)
                root.insert_left(node_val, node_index)
                createTree(root.left,left_node, right_node)
            else:
                root.insert_left(left[0][-1],None)
        if(len(right)==1):
            #print("$$$$$$$$$Right 0: ", right[0][-1])
            root.insert_right(right[0][-1],None)
        elif(len(right)>1):
            #Check if class is same in remaining subset
            label_set = np.unique(np.asarray(left)[:,-1])
            if(len(label_set>1)):
                node_index, node_val, left_node, right_node=Tree(np.asarray(right))
                print("Left: ",len(left_node))
                print("Right: ",len(right_node))
                #print("Index: ", node_index, node_val)
                root.insert_right(node_val, node_index)
                createTree(root.right,left_node, right_node)
            else:
                root.insert_right(right[0][-1],None) 
    return  


# In[145]:


# Function to calculate the performance metrics for the algorithm
def evaluation(predicted_value, actual_value):
    TP=TN=FP=FN=0
    for i in range(len(actual_value)):
        if(actual_value[i]==1 and predicted_value[i]==1):
            TP +=1
        elif(actual_value[i]==0 and predicted_value[i]==0):
            TN += 1
        elif(actual_value[i]==1 and predicted_value[i]==0):
            FN += 1
        elif(actual_value[i]==0 and predicted_value[i]==1):
            FP += 1
    print("TP:",TP,"TN:",TN,"FP:",FP,"FN:",FN)
    if (TP+TN+FP+FN)==0:
        accuracy=0
    else:
        accuracy=(TP+TN)/(TP+TN+FP+FN)
    if (TP+FP)==0:
        precision=0
    else:
        precision=(TP)/(TP+FP)
    if (TP+FN)==0:
        recall=0
    else:
        recall=(TP)/(TP+FN)
    if ((2*TP)+FN+FP)==0:
        f1=0
    else:
        f1=(2*TP)/((2*TP)+FN+FP)
    return accuracy, precision, recall, f1


# In[146]:


# Function to predict the label of test data from the decision tree created using training data
def predict(data):
    predict_label=[]
    for rows in data:
        #print(rows)
        n = root
        while(n):
            val = rows[n.index]
            #print(val,n.data,n.index)
            if val<n.data:
                prev=n
                n=n.left
            else:
                prev=n
                n=n.right
            if(n.index==None):
                prev=n
                break    
        predict_label.append(prev.data)
    return predict_label


# In[147]:


# Function to create K-Folds of the given dataset into test and train datasets
def create_fold(data,fold):
    #print(len(data))
    train = []
    test=[]
    if fold==1:
        #train=data
        n=(int)(len(data)*0.8)
        #print(n)
        train_fold = data.iloc[:n]
        test_fold = data.iloc[n:]
        train.append(train_fold)
        test.append(test_fold)
        #print(len(train), len(test))
    else:
        l = int(len(data)/fold)
        for i in range(fold):
            train_fold=data
            test_fold=[]
            for j in range(l):
                index = randrange(len(train_fold))
                #print(index)
                test_fold.append(train_fold.iloc[index])
                train_fold=train_fold.drop(train_fold.index[index])
            test.append(test_fold)
            train.append(train_fold)
    return test,train


# In[153]:


# Loop through all Folds, create Decision Tree and calculate performance metrics for each fold
if(choice==0):
    # Create folds for the given datasets
    #print(type(Data))
    #print(Data.shape)
    test_fold,train_fold = create_fold(Data,fold)
    accuracy=[]
    precision=[]
    recall=[]
    f1=[]
    i=1
    for f in range(fold):
        print("Fold Iteration: ",i)
        #train, test = splitData(f,0.80)
        #print("Shape: ",)
        train=train_fold[f]
        test=test_fold[f]
        print("Train Dataset: ",len(train))
        print("Test Dataset: ",len(test))
        train = np.asarray(train)
        test = np.asarray(test)
        root_index, root_val, left, right=Tree(train)
        print("Left: ",len(left))
        print("Right: ",len(right))
        root=Node(root_val,root_index)
        createTree(root,left,right)
        predicted_value=predict(test)
        actual_value=test[:,-1]
        a,p,r,f=evaluation(predicted_value, actual_value)
        if(a!=0):
            accuracy.append(a)
        if(p!=0):
            precision.append(p)
        if(r!=0):
            recall.append(r)
        if(f!=0):
            f1.append(f)
        i+=1
    # Calculate and print Average of all folds for each of the performance metrics
    print("Accuracy: ", mean(accuracy))
    print("Precision: ", mean(precision))
    print("Recall: ", mean(recall))
    print("F1 measure: ", mean(f1))
else:
    train=Data
    print("Train Dataset: ",len(train[0]))
    train = np.asarray(train)
    root_index, root_val, left, right=Tree(train)
    print("Left: ",len(left))
    print("Right: ",len(right))
    root=Node(root_val,root_index)
    createTree(root,left,right)
    root.display()

