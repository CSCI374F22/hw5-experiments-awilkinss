import sys
import csv
import sklearn
import pandas as pd
import numpy as np

# from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#reads in a dataset and returns the indexes of each non label column that has dtype object (string)
def one_hot_columns(data):
    indexes = []
    return_columns = []
    no_labels = data.drop('label',axis=1)
    all_types = list(no_labels.dtypes)

    for i in range(len(all_types)):
        if all_types[i] == 'object':
            indexes.append(i+1) #incremented because of truncated list(no label)
    
    #takes those indexes and finds the column names of string columns
    columns = list(data.columns)
    for i in range(len(columns)):
        if i in indexes:
            return_columns.append(columns[i])
    
    return return_columns

def encode(data,cols):
    #make all the hot encoded columns

    data = pd.get_dummies(data,columns=cols)

    #this will be converted to numpy arrays in the step before training
    return data

def hypo_models(file,rand_int):
#preprocessing
    data = pd.read_csv(file)
    cols = one_hot_columns(data)
    data = encode(data,cols)

    #training/testing split; x=attributes, y=labels
    x = data.drop("label",axis=1)
    y = data["label"]

    x = x.to_numpy()
    y = y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=rand_int, test_size=0.25, shuffle=True)

    #neural network init and fit
    nn = MLPClassifier(activation='logistic',solver='lbfgs',
    learning_rate='adaptive',early_stopping=True)

    nn.fit(x_train,y_train)

    #decision tree init and fit
    dt = DecisionTreeClassifier()
    dt.fit(x_train,y_train)

    #naive bayes init and fit
    nb = GaussianNB() #maybe add args, idk if i care
    nb.fit(x_train,y_train)

    #do predictions, validation, accuracy checking, matrix, etc. later
    return [[nn,dt,nb],[x_test,y_test]]

def mnist_models(file,rand_int):
    #preprocessing
    data = pd.read_csv(file)

    #training/testing split
    x = data.drop("label",axis=1)
    y = data["label"]

    x = x.to_numpy()
    y = y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=rand_int, test_size=0.25, shuffle=True)

    #neural network init and fit
    nn = MLPClassifier(activation='logistic',solver='lbfgs',
    learning_rate='adaptive',early_stopping=True)

    nn.fit(x_train,y_train)

    #decision tree init and fit
    dt = DecisionTreeClassifier()
    dt.fit(x_train,y_train)

    #naive bayes init and fit
    nb = GaussianNB() #maybe add args, idk if i care
    nb.fit(x_train,y_train)

    return [[nn,dt,nb],[x_test,y_test]]

def monks_models(file,rand_int):
    #preprocessing
    data = pd.read_csv(file)
    cols = one_hot_columns(data)
    data = encode(data,cols)

    #training/testing split
    x = data.drop("label",axis=1)
    y = data["label"]

    x = x.to_numpy()
    y = y.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=rand_int, test_size=0.25, shuffle=True)

    #neural network init and fit
    nn = MLPClassifier(activation='logistic',solver='lbfgs',
    learning_rate='adaptive',early_stopping=True)

    nn.fit(x_train,y_train)

    #decision tree init and fit
    dt = DecisionTreeClassifier()
    dt.fit(x_train,y_train)

    #naive bayes init and fit
    nb = GaussianNB() #maybe add args, idk if i care
    nb.fit(x_train,y_train)

    return [[nn,dt,nb],[x_test,y_test]]

def votes_models(file,rand_int):

    #preprocessing
    data = pd.read_csv(file)
    cols = one_hot_columns(data)
    data = encode(data,cols)

    #training/testing split
    x = data.drop("label",axis=1)
    y = data["label"]

    x = x.to_numpy()
    y = y.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=rand_int, test_size=0.25, shuffle=True)

    #neural network init and fit
    nn = MLPClassifier(activation='logistic',solver='lbfgs',
    learning_rate='adaptive',early_stopping=True)

    nn.fit(x_train,y_train)

    #decision tree init and fit
    dt = DecisionTreeClassifier()
    dt.fit(x_train,y_train)

    #naive bayes init and fit
    nb = GaussianNB() #maybe add args, idk if i care
    nb.fit(x_train,y_train)

    return [[nn,dt,nb],[x_test,y_test]]

def test():
    data = pd.read_csv('hypothyroid.csv')
    cols = one_hot_columns(data)
    print("cols:",cols)
    result = encode(data,cols)
    # print("result:")
    # print(result)

file_names = ['hypothyroid.csv','mnist_1000.csv','monks1.csv','votes.csv']

def main():
    rand_int = int(sys.argv[1])

    for file in file_names:
        #model lists= nn(neural network), dt(decision tree), and nb(naive bayes)

        #testing lists are x_test and y_test

        #models are not, in any way, optimized
        if file == 'hypothyroid.csv':
            model_list, testing_lists = hypo_models(file,rand_int)

        if file == 'mnist_1000.csv':
            model_list, testing_lists = mnist_models(file,rand_int)
            

        if file == 'monks1.csv':
            model_list, testing_lists = monks_models(file,rand_int)
        
        if file == 'votes.csv':
            model_list, testing_lists = votes_models(file,rand_int)

    # test()

main()

#Goal Flow:

#1. Read in one file at a time from file_names list
#2. Do Preprocessing (find if it has nominal attributes)
#3. split into training and testing
#4. Fit one of the three models based off its nominal_scan return
#5. find accuracy and build confusion matrix of each model
#6. Output accuracy and matrix as a file
#7. Repeat with next file

#forseeable issues:

#dataframes are slow, convert to numpy after preprocessing
#not all models work with nominal attributes, preprocessing has to respond accordingly and tell them which models to build
#data sets with both will have to one hot encode nominal columns
#mnist_1000 is the only purely numeric dataset

#model contingencies:

#neural networks: MLPRegressor or MLPClassifier
#Decision Trees: Cart works with numeric (regression), and regular works for classification()
#naive bayes: only classification; we'll see how this works out using GaussianNP
