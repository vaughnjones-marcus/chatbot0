# -*- coding: utf-8 -*-
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json


with open("intents.json") as file:
    data = json.load(file)


words = []
labels = []
docs_x = []
docs_y =[]

#docs x and y for each pattern y holds intent tag

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        #nkl tokenizes the words aka splits them up
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
        #each entry in x x corrresponds to an entry in docs y x pattern intent in y
        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            
words = [stemmer.stem(w.lower()) for w in words if  w !="?"]
words = sorted(list(set(words))) #takes out duplicats 

labels = sorted(labels)

#bag of words onehot encoded list the lenght of the words we have tellign if a word exist 

training = []
output  = []

out_empty = [0 for _ in range(len(labels))]


for x, doc in enumerate(docs_x):
    bag = []
    
    wrds = [stemmer.stem(w) for w in doc]
   #adding a 1 or a zero if the word exists or not 
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    #looking through labels list finds the tag in the list and set the output to one in the row
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    #two list traingni list and outputs in zeros and ones
    training.append(bag)
    output.append(output_row)
    
training = numpy.array(training)
output = numpy.array(output)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
#fully connected layer added to neural netwoprk 8 neurons on the hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# softmax gives probability of each neuron in the layer and thats the output
net = tflearn.fully_connected(net, len(output[0]), activation= "softmax")
net = tflearn.regression(net)

#to train the model DNN deep neural network an ANN with multipple hidden lauers
model = tflearn.DNN(net)
#epoch is the amount of time it will see the same data
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")



"""


"""
           





#3:00 2
            
        




"""
Created on Sat Apr  1 15:40:24 2023

@author: marcus
"""

