import sys
import csv
import sklearn
import pandas as pd
import numpy as np
import math

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

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

#gets possible labels by taking all the values in the label column and adding them to a set
def get_possible_labels(data):
    possible = []
    all_of_them = list(data["label"])
    scanna = set()

    for label in all_of_them:
        scanna.add(label)
    
    return list(scanna)

#returns a list of fitted models, testing and training lists, and the possible labels for the hypothyroid data set

def hypo_models(file,rand_int):
    #preprocessing
    data = pd.read_csv(file)
    cols = one_hot_columns(data)
    data = encode(data,cols)

    #get the possible labels
    possible_labels = get_possible_labels(data)

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
    return [[nn,dt,nb],[x_test,y_test],possible_labels]

#returns a list of fitted models, testing and training lists, and the possible labels for the mnist_1000 data set

def mnist_models(file,rand_int):
    #preprocessing
    data = pd.read_csv(file)

    #training/testing split
    x = data.drop("label",axis=1)
    y = data["label"]

    #get the possible labels
    possible_labels = get_possible_labels(data)

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

    return [[nn,dt,nb],[x_test,y_test],possible_labels]

#returns a list of fitted models, testing and training lists, and the possible labels for the monks data set
def monks_models(file,rand_int):
    #preprocessing
    data = pd.read_csv(file)
    cols = one_hot_columns(data)
    data = encode(data,cols)

    #get the possible labels
    possible_labels = get_possible_labels(data)

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

    return [[nn,dt,nb],[x_test,y_test],possible_labels]

#returns a list of fitted models, testing and training lists, and the possible labels for the votes data set

def votes_models(file,rand_int):

    #preprocessing
    data = pd.read_csv(file)
    cols = one_hot_columns(data)
    data = encode(data,cols)

    #get the possible labels
    possible_labels = get_possible_labels(data)

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

    return [[nn,dt,nb],[x_test,y_test],possible_labels]

#takes the arguements and gives a cool lil file name
def gen_file_name(model,file,rand_seed):
    #results-<Approach>-<DataSet>_<Seed>.csv
    if type(model) == sklearn.neural_network.MLPClassifier:
        file_name = 'results-' + 'NeuralNetwork-' + file.replace('.csv','') + '-' + str(rand_seed)

    elif type(model) == sklearn.tree.DecisionTreeClassifier:
        file_name = 'results-' + 'DecisionTree-' + file.replace('.csv','') + '-' + str(rand_seed)
    
    else:
        file_name = 'results-' + 'NaiveBayes-' + file.replace('.csv','') + '-' + str(rand_seed)
    
    return file_name

#returns a string of the model name for pretty printing
def gen_quality_of_life(model):
    if type(model) == sklearn.neural_network.MLPClassifier:
        model_name = 'Neural-Network'

    elif type(model) == sklearn.tree.DecisionTreeClassifier:
        model_name = 'Decision-Tree'
    
    else:
        model_name = 'Naive-Bayes'
    
    return model_name

#prints the matrix as a file using the aforementioned 'cool lil file name'
def matrix_print(matrix,possible_labels,file_name):
    header = []
    for label in possible_labels:
        header.append(label)

    with open(str(file_name), 'w') as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(list(header))

        for i in range(len(matrix)):
            row = np.append(matrix[i],(possible_labels[i]),axis=None)
            writer.writerow(row)
    file.close

#returns accuracy and confusion matrix in a list
def validate(predictions,actual,possible_labels):
    accuracy = 0
    correct = 0
    length = len(actual)

    #if the prediction was right, increment count by 1
    for i in range(length):
        if predictions[i] == actual[i]:
            correct += 1

    #get accuracy by finding percent it got correct
    accuracy = correct / length

    #generate confusion matrix

    # print("predictions:",predictions)
    # print("actual(labels",actual)
    # print("possible labels:",possible_labels)

    matrix = confusion_matrix(actual,predictions,labels=possible_labels)

    return [matrix,accuracy]

def test():
    data = pd.read_csv('hypothyroid.csv')
    cols = one_hot_columns(data)
    print("cols:",cols)
    result = encode(data,cols)
    # print("result:")
    # print(result)

def confidence_interval(matrix):
    #Get n and p_hat values from matrix
    n = 0
    p_hat = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            #Count total number of values in tested
            n += matrix[i][j]

            #Count number of correct predictions
            if i == j:
                p_hat += matrix[i][j]
    
    #Get all values needed to calculate CI
    p_hat = p_hat / n
    s = math.sqrt(p_hat * (1 - p_hat))
    SE = s / math.sqrt(n)
    z = 2.24

    #Calculate CI
    lower_bound = round(p_hat - z * SE, 4)
    upper_bound = round(p_hat + z * SE, 4)
    return tuple([lower_bound, upper_bound])

file_names = ['hypothyroid.csv','mnist_1000.csv','monks1.csv','votes.csv']

def main():
    rand_int = int(sys.argv[1])

    for file in file_names:
        #model lists= nn(neural network), dt(decision tree), and nb(naive bayes)

        #testing lists are x_test and y_test

        #models are not, in any way, shape, or form, optimized
        if file == 'hypothyroid.csv':
            model_list, testing_lists, labels = hypo_models(file,rand_int)

            #once each model is fit, generate predictions, get accuracy, etc. for each model

            print("=== Hypothyroid Data Set ===")
            for model in model_list:
                #gets the name for the to be generated file
                file_name = gen_file_name(model,file,rand_int)

                #generates predictions
                predictions = model.predict(testing_lists[0])

                #gets the conf. matrix and the accuracy of each model
                matrix, accuracy = validate(predictions,testing_lists[1],labels)

                #print accuracy (on command line) then matrix (as file)
                CI = confidence_interval(matrix)
                print("Accuracy for",gen_quality_of_life(model),':', accuracy, ", CI: ", CI)
                matrix_print(matrix,labels,file_name)


        if file == 'mnist_1000.csv':
            model_list, testing_lists, labels = mnist_models(file,rand_int)

            print("=== mnist_1000 Data Set ===")
            for model in model_list:
                #gets the name for the to be generated file
                file_name = gen_file_name(model,file,rand_int)

                #generates predictions
                predictions = model.predict(testing_lists[0])

                #gets the conf. matrix and the accuracy of each model
                matrix, accuracy = validate(predictions,testing_lists[1],labels)

                #print accuracy (on command line) then matrix (as file)
                CI = confidence_interval(matrix)
                print("Accuracy for",gen_quality_of_life(model),':', accuracy, ", CI: ", CI)
                matrix_print(matrix,labels,file_name)
            

        if file == 'monks1.csv':
            model_list, testing_lists, labels = monks_models(file,rand_int)

            print("=== Monks Data Set ===")
            for model in model_list:
                #gets the name for the to be generated file
                file_name = gen_file_name(model,file,rand_int)

                #generates predictions
                predictions = model.predict(testing_lists[0])

                #gets the conf. matrix and the accuracy of each model
                matrix, accuracy = validate(predictions,testing_lists[1],labels)

                #print accuracy (on command line) then matrix (as file)
                CI = confidence_interval(matrix)
                print("Accuracy for",gen_quality_of_life(model),':', accuracy, ", CI: ", CI)
                matrix_print(matrix,labels,file_name)
        
        if file == 'votes.csv':
            model_list, testing_lists, labels = votes_models(file,rand_int)

            print("=== Votes Data Set ===")
            for model in model_list:
                #gets the name for the to be generated file
                file_name = gen_file_name(model,file,rand_int)

                #generates predictions
                predictions = model.predict(testing_lists[0])

                #gets the conf. matrix and the accuracy of each model
                matrix, accuracy = validate(predictions,testing_lists[1],labels)

                #print accuracy (on command line) then matrix (as file)
                CI = confidence_interval(matrix)
                print("Accuracy for",gen_quality_of_life(model),':', accuracy, ", CI: ", CI)
                matrix_print(matrix,labels,file_name)

    # test()

# main()

def run(rand_int):
    hypo_accuracies = [[],[],[]]
    mnist_accuracies = [[],[],[]]
    monks_accuracies = [[],[],[]]
    votes_accuracies = [[],[],[]]

    for file in file_names:
        #model lists= nn(neural network), dt(decision tree), and nb(naive bayes)

        #testing lists are x_test and y_test

        #models are not, in any way, shape, or form, optimized
        if file == 'hypothyroid.csv':
            model_list, testing_lists, labels = hypo_models(file,rand_int)

            #once each model is fit, generate predictions, get accuracy, etc. for each model

            print("=== Hypothyroid Data Set ===")
            for model in model_list:
                #gets the name for the to be generated file
                file_name = gen_file_name(model,file,rand_int)

                #generates predictions
                predictions = model.predict(testing_lists[0])

                #gets the conf. matrix and the accuracy of each model
                matrix, accuracy = validate(predictions,testing_lists[1],labels)

                #print accuracy (on command line) then matrix (as file)
                CI = confidence_interval(matrix)
                print("Accuracy for",gen_quality_of_life(model),':', accuracy, ", CI: ", CI)
                # matrix_print(matrix,labels,file_name)

                if type(model) == sklearn.neural_network.MLPClassifier:
                    hypo_accuracies[0].append(accuracy)
                elif type(model) == sklearn.tree.DecisionTreeClassifier:
                    hypo_accuracies[1].append(accuracy)
                else:
                    hypo_accuracies[2].append(accuracy)

        if file == 'mnist_1000.csv':
            model_list, testing_lists, labels = mnist_models(file,rand_int)

            print("=== mnist_1000 Data Set ===")
            for model in model_list:
                #gets the name for the to be generated file
                file_name = gen_file_name(model,file,rand_int)

                #generates predictions
                predictions = model.predict(testing_lists[0])

                #gets the conf. matrix and the accuracy of each model
                matrix, accuracy = validate(predictions,testing_lists[1],labels)

                #print accuracy (on command line) then matrix (as file)
                CI = confidence_interval(matrix)
                print("Accuracy for",gen_quality_of_life(model),':', accuracy, ", CI: ", CI)
                # matrix_print(matrix,labels,file_name)

                if type(model) == sklearn.neural_network.MLPClassifier:
                    mnist_accuracies[0].append(accuracy)
                elif type(model) == sklearn.tree.DecisionTreeClassifier:
                    mnist_accuracies[1].append(accuracy)
                else:
                    mnist_accuracies[2].append(accuracy)
         
            

        if file == 'monks1.csv':
            model_list, testing_lists, labels = monks_models(file,rand_int)

            print("=== Monks Data Set ===")
            for model in model_list:
                #gets the name for the to be generated file
                file_name = gen_file_name(model,file,rand_int)

                #generates predictions
                predictions = model.predict(testing_lists[0])

                #gets the conf. matrix and the accuracy of each model
                matrix, accuracy = validate(predictions,testing_lists[1],labels)

                #print accuracy (on command line) then matrix (as file)
                CI = confidence_interval(matrix)
                print("Accuracy for",gen_quality_of_life(model),':', accuracy, ", CI: ", CI)
                # matrix_print(matrix,labels,file_name)
                if type(model) == sklearn.neural_network.MLPClassifier:
                    monks_accuracies[0].append(accuracy)
                elif type(model) == sklearn.tree.DecisionTreeClassifier:
                    monks_accuracies[1].append(accuracy)
                else:
                    monks_accuracies[2].append(accuracy)
        
        if file == 'votes.csv':
            model_list, testing_lists, labels = votes_models(file,rand_int)

            print("=== Votes Data Set ===")
            for model in model_list:
                #gets the name for the to be generated file
                file_name = gen_file_name(model,file,rand_int)

                #generates predictions
                predictions = model.predict(testing_lists[0])

                #gets the conf. matrix and the accuracy of each model
                matrix, accuracy = validate(predictions,testing_lists[1],labels)

                #print accuracy (on command line) then matrix (as file)
                CI = confidence_interval(matrix)
                print("Accuracy for",gen_quality_of_life(model),':', accuracy, ", CI: ", CI)
                # matrix_print(matrix,labels,file_name)

                if type(model) == sklearn.neural_network.MLPClassifier:
                    votes_accuracies[0].append(accuracy)
                elif type(model) == sklearn.tree.DecisionTreeClassifier:
                    votes_accuracies[1].append(accuracy)
                else:
                    votes_accuracies[2].append(accuracy)  

    for lst in [hypo_accuracies,mnist_accuracies,monks_accuracies,votes_accuracies]:
        print(lst)

    return [hypo_accuracies,mnist_accuracies,monks_accuracies,votes_accuracies]

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
