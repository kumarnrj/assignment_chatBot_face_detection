import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer =LancasterStemmer()

import os
import time 
import playsound
import speech_recognition as sr
from gtts import gTTS
import numpy as np
import tflearn
import random
import json
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
import pickle as pk
with open("intents.json") as file:
    data = json.load(file)
try:
    with open('data.pickle','rb') as f:
        words,labels,training,output = pk.load(f)

except:
    
    words =  []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
    
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    
    labels = sorted(labels)
    
    # bag of words
    training = []
    output = []
    out_empty = [ 0 for _ in range(len(labels))]
     
    for x, doc in enumerate( docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] =1
        
        training.append(bag)
        output.append(output_row)
        
    training = np.array(training)
    output = np.array(output)
    
#    with open('data.pickle','wb') as f:
 #       pk.dump((words,labels,training,output),f)

ops.reset_default_graph()

net = tflearn.input_data(shape =[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation = 'softmax')
net =tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training,output,n_epoch = 1000, batch_size = 8, show_metric = True)
'''
try:
    model.load('model.tflearn')
except:
    model.fit(training,output,n_epoch = 1000, batch_size = 8, show_metric = True)
    model.save("model.tflearn")
        
 '''       

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [ stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i] =1
                
    return np.array(bag)


def speak(text,count):
    voice = gTTS(text=text,lang="en")
    filename ="voice"+str(count)+".mp3"
    print(filename)
    voice.save(filename)
    playsound.playsound(filename)
    
def chat():
    print(" Start talking with the bot (type quit to stop)")
    count =0
    while(True):
        inp = input("You :")
        if inp.lower() == "quit":
            break
        result = model.predict([bag_of_words(inp,words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]
        
        if result[result_index] >0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            count=count+1
            #print(random.choice(responses))
            speak(random.choice(responses),count)
            
        else:
            print(" i did'nt get that plz try again")




chat()

