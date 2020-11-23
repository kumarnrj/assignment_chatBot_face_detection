# Project Description 
The project is chat bot and it created using NLP (natural Languange Processing), deep learning and gTTS (google text to speach).

## What is NLP?
[NLP](https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1) is a field of Artificial Intelligence that gives the machines the ability to read, understand and derive meaning from human languages.
It is a discipline that focuses on the interaction between data science and human language, and is scaling to lots of industries. Today NLP is booming thanks to the huge improvements in the access to data and the increase in computational power, which are allowing practitioners to achieve meaningful results in areas like healthcare, media, finance and human resources, among others.
## What is Deep Learning?
[Deep Learning](https://machinelearningmastery.com/what-is-deep-learning/) is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.

## What is gTTS?
[gTTS](https://www.geeksforgeeks.org/convert-text-speech-python/) There are several APIs available to convert text to speech in python. One of such APIs is the Google Text to Speech API commonly known as the gTTS API. gTTS is a very easy to use tool which converts the text entered, into audio which can be saved as a mp3 file.

The gTTS API supports several languages including English, Hindi, Tamil, French, German and many more. The speech can be delivered in any one of the two available audio speeds, fast or slow. However, as of the latest update, it is not possible to change the voice of the generated audio.

## Get Started
---
### Installation
``` python
>>> pip install tensorflow
>>> pip install gTTS
>>> pip install playsound
>>> pip install pyaudio
>>> pip install SpeechRecognition
>>> pip install numpy
>>> pip install tflearn
>>> pip install ntlk
```
### Import the necessary Libraries.
``` python
>>> import nltk
>>> from nltk.stem.lancaster import LancasterStemmer
>>> stemmer =LancasterStemmer()

>>> import os
>>> import time 
>>> import playsound
>>> import speech_recognition as sr
>>> from gtts import gTTS
>>> import numpy as np
>>> import tflearn
>>> import random
>>> import json
>>> import tensorflow as tf
>>> from tensorflow.python.framework import ops
```
### What is nltk?
[NLTK](https://www.nltk.org/) is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

### what is the use of LancasterStemmer()
Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language.
The LancasterStemmer (Paice-Husk stemmer) is an iterative algorithm with rules saved externally. One table containing about 120 rules indexed by the last letter of a suffix. On each iteration, it tries to find an applicable rule by the last character of the word. Each rule specifies either a deletion or replacement of an ending. If there is no such rule, it terminates. It also terminates if a word starts with a vowel and there are only two letters left or if a word starts with a consonant and there are only three characters left. Otherwise, the rule is applied, and the process repeats.

More details read [Lancaster](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python)

# Code Description
``` python
# Opening the json file with contain patterns and responses based on these pattern and responses our chat bot will response.
>>> with open("intents.json") as file:
>>>    data = json.load(file)
```
Storing the root words and lebels(greeting,bye etc.) in the the list. 
```python
>>> words =  []
>>> labels = []
>>> docs_x = []
>>> docs_y = []
    
>>> for intent in data['intents']:
>>>     for pattern in intent['patterns']:
>>>         wrds = nltk.word_tokenize(pattern)
>>>         words.extend(wrds)
>>>         docs_x.append(wrds)
>>          docs_y.append(intent["tag"])
            
>>>       if intent["tag"] not in labels:
>>>           labels.append(intent["tag"])
```
removing the duplicate from the list and sorting the data.

``` python
>>> words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
>>>    words = sorted(list(set(words)))
    
>>>    labels = sorted(labels)
```
Creating the bag of words and trainig our model.

``` python
>>>    # bag of words
>>>    training = []
>>>    output = []
>>>    out_empty = [ 0 for _ in range(len(labels))]
     
>>>    for x, doc in enumerate( docs_x):
>>>        bag = []
>>>        wrds = [stemmer.stem(w) for w in doc]
        
>>>        for w in words:
>>>            if w in wrds:
>>>                bag.append(1)
>>>            else:
>>>                bag.append(0)
>>>        output_row = out_empty[:]
>>>        output_row[labels.index(docs_y[x])] =1
        
>>>        training.append(bag)
>>>        output.append(output_row)
        
>>>    training = np.array(training)
>>>    output = np.array(output)
    
#    with open('data.pickle','wb') as f:
 #       pk.dump((words,labels,training,output),f)

>>> ops.reset_default_graph()

>>> net = tflearn.input_data(shape =[None,len(training[0])])
>>> net = tflearn.fully_connected(net,8)
>>> net = tflearn.fully_connected(net,8)
>>> net = tflearn.fully_connected(net,len(output[0]),activation = 'softmax')
>>> net =tflearn.regression(net)

>>> model = tflearn.DNN(net)
>>> model.fit(training,output,n_epoch = 1000, batch_size = 8, show_metric = True) 
```
---
### What is tflearn?
[TFlearn](http://tflearn.org/) is a modular and transparent deep learning library built on top of Tensorflow. It was designed to provide a higher-level API to TensorFlow in order to facilitate and speed-up experimentations, while remaining fully transparent and compatible with it.

---
Bag_of_word method
``` python
>>> def bag_of_words(s,words):
>>>    bag = [0 for _ in range(len(words))]
    
>>>    s_words = nltk.word_tokenize(s)
>>>    s_words = [ stemmer.stem(word.lower()) for word in s_words]
    
>>>    for se in s_words:
>>>        for i,w in enumerate(words):
>>>            if w == se:
>>>                bag[i] =1
                
>>>    return np.array(bag)
```

### Speak method 
It will recieve a text as parameter and with the help of gTTS it will process the text and convert it into audio file and in order to use it we have to save it and then with the help of soundplayer() it will process the audio.

``` python
>>>def speak(text,count):
>>>    voice = gTTS(text=text,lang="en")
>>>    filename ="voice"+str(count)+".mp3"
>>>    print(filename)
>>>    voice.save(filename)
>>>    playsound.playsound(filename)
```

### Chat method
It will intract with the user.

``` python
>>> def chat():
>>>     print(" Start talking with the bot (type quit to stop)")
>>>     count =0
>>>     while(True):
>>>        inp = input("You :")
>>>        if inp.lower() == "quit":
            break
>>>        result = model.predict([bag_of_words(inp,words)])[0]
>>>        result_index = np.argmax(result)
>>>        tag = labels[result_index]
        
>>>        if result[result_index] >0.7:
>>>            for tg in data['intents']:
>>>                if tg['tag'] == tag:
>>>                    responses = tg['responses']
>>>           count=count+1
>>>            #print(random.choice(responses))
>>>            speak(random.choice(responses),count)
            
>>>        else:
>>>            print(" i did'nt get that plz try again")
>>> chat()
```



