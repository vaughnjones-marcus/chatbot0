# -*- coding: utf-8 -*-
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle


with open("intents.json") as file:
    data = json.load(file)

'''
to not run through the porgram aftrer every 
question we try to open up saved data if it
doesnt work then it will run through
'''
try:
    #put x here when i change the json file so it doesnt open old pickle data or delete pickle file 
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
        
    
    
except:
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
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

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

try:
    model.load("model.tflearn")
except:     
    #epoch is the amount of time it will see the same data
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return numpy.array(bag)

def chat():
    print("Please talk to M (type quit when you done)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    
            print(random.choice(responses))
        else:
            print("Im sorry im a little confused could you rephrase that")
            
        
        
   
    '''
Anyone that finds any problems in retraining their data
 (does not respond to newly added intents) just simply 
 delete the lines of code that relates to the loading 
 of Pickle file since your program runs on that source
 file, so you'll need to delete it to retrain your 
 program with the new intents. Just make sure to save
 it again in the Pickle file if your chat bot is ready
 so it won't retrain again when its on use.
'''    
        
        
chat()

    

           





#3:00 2
            
        




"""
Created on Sat Apr  1 15:40:24 2023

@author: marcus
"""

