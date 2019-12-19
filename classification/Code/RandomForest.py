#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import sys
from statistics import mean
from random import randrange
from random import seed
from sklearn import preprocessing
import random


# In[2]:


# Data structure to save tree where each node contains values: split value and index of split value
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
            line = '%s' % self.data#+"   "+str(self.index)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.data#+"   "+str(self.index)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.data#+"   "+str(self.index)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.data#+"   "+str(self.index)
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


# In[3]:


# User Input for Dataset, Number of Folds & Number of Trees to be created in each fold
file_name = input("Enter file name: ")
fold = int(input("Enter number of folds: "))
n_trees=int(input("Enter number of Trees: "))
option=int(input("Enter 1 for selecting number of features else 0: "))
if(option==1):
    n_features=int(input("Enter number of Features to be selected for each Tree: "))


# In[4]:


# Reading file into Dataframe
Data = pd.read_csv(file_name, sep='\t', lineterminator='\n', header=None)


# In[5]:


#Data.head


# In[6]:


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


# In[7]:


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


# In[8]:


# Check if the dataset has any string values and converting the string values to its one hot encoding values
flag, str_index = check_string(Data)
np_data = Data.to_numpy()
#print(flag, str_index)
for i in str_index:
    np_data[:,i]=one_hot_encoding(np_data[:,i])
#np_data[0]
Data=pd.DataFrame(np_data)
#Data


# In[9]:


print(len(Data))
#print(Data.loc[0])


# In[10]:


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


# In[11]:


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


# In[12]:


# Function to get the minimum GINI Index value, its index, left and right subsets
def Tree(data,n_features):
    #print("data to create Tree: ",data.shape)
    unique_label = np.unique(data[:,-1])
    col = len(data[0])-1
    rows = len(data)
    min_gini = sys.float_info.max
    index = 0
    left_tree=[]
    right_tree=[]
    value = 0
    cols = random.sample(range(col), n_features)
    #print("Features selected: ",cols)
    for c in cols:
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
    return  index, value, left_tree, right_tree


# In[13]:


# Function to create Tree based on root, left and right subsets.
def createTree(root,left,right,n_features):
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
                node_index, node_val, left_node, right_node=Tree(np.asarray(left),n_features)
                #print("Index: ", node_index, node_val)
                root.insert_left(node_val, node_index)
                createTree(root.left,left_node, right_node,n_features)
            else:
                root.insert_left(left[0][-1],None)
        if(len(right)==1):
            #print("$$$$$$$$$Right 0: ", right[0][-1])
            root.insert_right(right[0][-1],None)
        elif(len(right)>1):
            #Check if class is same in remaining subset
            label_set = np.unique(np.asarray(left)[:,-1])
            if(len(label_set>1)):
                node_index, node_val, left_node, right_node=Tree(np.asarray(right),n_features)
                #print("Index: ", node_index, node_val)
                root.insert_right(node_val, node_index)
                createTree(root.right,left_node, right_node,n_features)
            else:
                root.insert_right(right[0][-1],None) 
    return  


# In[14]:


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


# In[15]:


# Function to predict the label of test data from the decision tree created using training data
def predict(data,root):
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


# In[16]:


# Function to create K-Folds of the given dataset into test and train datasets
def create_fold(data,fold):
    #print(len(data))
    train = []
    test=[]
    if fold==1:
        #train=data
        n=(int)(len(data)*0.8)
        #print(n)
        train_fold = data.iloc[:n]#indexes rows for training data
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


# In[17]:


# Function to create bagging dataset from training dataset
def create_bags(data,n_trees):
    bag=[]
    #print(type(data), len(data))
    for i in range(n_trees):
        b=[]
        for j in range(len(data)):
            index = randrange(len(data))
            #print("Index:", index, len(data))
            #print("Data: ",data[index])
            b.append(data.iloc[index])
        bag.append(b)
    return bag


# In[18]:


# Create folds for the given datasets
#print(type(Data))
#print(Data.shape)
test_fold,train_fold = create_fold(Data,fold)


# In[19]:


# Loop through all Folds and Bags, create Decision Tree and calculate performance metrics for each fold
accuracy=[]
precision=[]
recall=[]
f1=[]
for f in range(fold):
    print("Fold Iteration: ", f+1)
    train = train_fold[f]
    test = test_fold[f]
    test = np.asarray(test)
    bags = create_bags(train, n_trees)
    #print("Train",len(train.iloc[0]), type(train))
    if(option==0):
        n_features = int(math.sqrt(len(train.iloc[0])))
        #n_features = int(len(train.iloc[0])/5)
    print("No. of features selected:", n_features)
    trees = []
    predicted = []
    actual_value=test[:,-1]
    i=1
    for bag in bags:
        print("Bag iteration: ", i)
        train_bag = np.asarray(bag)
        root_index, root_val, left, right=Tree(train_bag,n_features)
        root=Node(root_val,root_index)
        createTree(root,left,right,n_features)
        pred = predict(test,root)
        predicted.append(pred)
        trees.append(root)
        i+=1
    predicted_value=[]
    #Selecting the Majority voting for predicted label among all trees
    for p in range(len(predicted[0])):
        #p_label=predicted[:,p]
        p_label=[int(row[p]) for row in predicted]
        max_pred = max(p_label,key=p_label.count)
        #print(type(p_label))
        #print(type(max_pred))
        print("Predicted List: ",p_label , "\tMajority Value: ",int(max_pred))
        predicted_value.append(max_pred)
    a,p,r,f=evaluation(predicted_value, actual_value)
    if(a!=0):
        accuracy.append(a)
    if(p!=0):
        precision.append(p)
    if(r!=0):
        recall.append(r)
    if(f!=0):
        f1.append(f)
print("Accuracy: ", mean(accuracy))
print("Precision: ", mean(precision))
print("Recall: ", mean(recall))
print("F1 measure: ", mean(f1))       

