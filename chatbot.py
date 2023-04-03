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
        docs_x.appends(pattern)
        docs_y.append(intent["tag"])
        
        #each entry in x x corrresponds to an entry in docs y x pattern intent in y
        
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words))) #takes out duplicats 

labels = sorted(labels)

#3:00 2
            
        




"""
Created on Sat Apr  1 15:40:24 2023

@author: marcus
"""

