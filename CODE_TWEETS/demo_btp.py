from textblob import TextBlob
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input, LSTM
from keras.models import Sequential, Model
from keras.layers import SimpleRNN, Activation, Dense, Dropout, Embedding, Flatten, Input, Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Conv1D, Concatenate
import numpy as np
from nltk import tokenize
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import KFold
from keras.utils import np_utils
from string import punctuation
import codecs
import operator
import gensim, sklearn
from collections import defaultdict
import sys
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy
import os
import io
import re
import json
import csv
import pandas
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from empath import Empath
from keras.layers.merge import concatenate
from gensim.models import Word2Vec
from keras.models import load_model


print "check 1"

GLOVE_MODEL_FILE='/home/raghav/Documents/BTP/TWEETS/EMBEDDINGS/glove.twitter.27B.100d.txt'
SEED = 42

word2vec_output_file = '/home/raghav/Documents/BTP/TWEETS/EMBEDDINGS/glove.6B.100d.txt.word2vec'
glove2word2vec(GLOVE_MODEL_FILE, word2vec_output_file)
filename='/home/raghav/Documents/BTP/TWEETS/EMBEDDINGS/glove.6B.100d.txt.word2vec'

#filename='/home/raghav/Documents/Dvesh-Prahari/Code/FastText/wiki.en.vec'
word2vec_model = KeyedVectors.load_word2vec_format(filename, binary=False)
np.random.seed(SEED)


while(1):

    EMBEDDING_DIMENSION=100
    LOSS_FUNCTION='categorical_crossentropy'
    OPTIMIZER='adam'
    EPOCHS=10
    BATCH_SIZE=128
    FOLDS=10
    INITIALIZE_WEIGHTS='random'

    print("Enter Tweet : ")
    tweet_input = str(raw_input())

    row = ['0', tweet_input]

    with open('/home/raghav/Documents/BTP/TWEETS/DATASET/HEOT/HOT_Dataset_modified.csv', 'r') as readFile:
        reader = csv.reader(readFile)
        lines = list(reader)
        lines[0] = row

    with open('/home/raghav/Documents/BTP/TWEETS/DATASET/HEOT/HOT_Dataset_modified.csv', 'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(lines)



    vocab={}
    reverse_vocab={}
    freq = defaultdict(int)
    vocab_index = 1

    FLAGS = re.MULTILINE | re.DOTALL

    print "check 2"

    PV={}
    PV_location={}

    lexicon = Empath()
    with open('/home/raghav/Documents/BTP/TWEETS/DATASET/Hinglish_Profanity_List.csv', 'r') as myfile:
        reader = csv.reader(myfile , delimiter=',')
        i=0
        for row in reader:
            #print row
            PV[row[0]]=int(row[2])
            PV[row[1]]=int(row[2])
            PV_location[row[0]]=i
            PV_location[row[1]]=i
            i=i+1

    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = u"<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = " ".join(["<hashtag>"])
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"


    def tokenize_glove_func(text):
      
        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=FLAGS)

        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ")
        text = re_sub(r"/"," ")
        text = re_sub(r"@\w+", " ")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " ")
        text = re_sub(r"{}{}p+".format(eyes, nose), " ")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " ")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " ")
        text = re_sub(r"<3","<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " ")
        text = re_sub(r"#\S+",  " ")
        text = re_sub(r"([!?.]){2,}", r" ")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r" ")
        text = re_sub(r"([A-Z]){2,}"," ")

        return text.lower()

    def glove_tokenize(text):
       
        global vocab_index
        
        text = tokenize_glove_func(text)
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        temp_pv=[]
        for i in range(210):
            temp_pv.append(0)
        for word in words:

                try:
                    temp_pv[PV_location[word]]=PV[word]
                except:
                    pass

                if word not in vocab:
                    vocab[word] = vocab_index
                    reverse_vocab[vocab_index] = word      
                    vocab_index += 1
                freq[word] += 1
        

        return words, temp_pv


    def glove_tokenize_transfer_learning(text,hinglish_dictionary,count_Hinglish,count_English,count_total):
        global vocab_index
        text = tokenize_glove_func(text)
        
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        count_total=count_total+len(words)
        words = [word for word in words if word not in STOPWORDS]
        
        temp_pv=[]
        for i in range(210):
            temp_pv.append(0)
        for word in words:
            try:
                temp=hinglish_dictionary[word]
                temp = temp.encode('utf-8')
                count_Hinglish=count_Hinglish+1
            except:
                temp=word
                count_English=count_English+1
            words[words.index(word)]=temp
            #print type(words[words.index(word)])
            #print type(temp)

            try:
                temp_pv[PV_location[temp]]=PV[temp]
            except:
                pass       


        for word in words:
                if word not in vocab:
                    vocab[word] = vocab_index
                    reverse_vocab[vocab_index] = word      
                    vocab_index += 1
                freq[word] += 1
        

        return words,count_Hinglish,count_English,count_total, temp_pv


    data={}
    data['tweets']=[]
    f=io.open('/home/raghav/Documents/BTP/TWEETS/DATASET/Davidson/labeled_data.csv', newline='')
    reader=csv.reader(f)

    for row in reader:
        data['tweets'].append({      
            'text'     : row[6],        ########
            'label'    : row[5]         ########
        })

    trans_data={}
    trans_data['tweets']=[]
    count_non=0
    count_abuse=0
    count_hate=0
    with open('/home/raghav/Documents/BTP/TWEETS/DATASET/HEOT/HOT_Dataset_modified.csv', 'r') as myfile:
        reader = csv.reader(myfile , delimiter=',')
        for row in reader:
            if(len(row)<2):  ##########
                continue
            
            if row[0]== '0':
                count_non=count_non+1
            
            if row[0]== '1':
                count_hate=count_hate+1
            
            if row[0]== '2':
                count_abuse=count_abuse+1
            
            trans_data['tweets'].append({

            
                'text'  : row[1],  #########
                'label' : row[0]   #########


        })  


    jsonfile_transliteration = open('/home/raghav/Documents/BTP/TWEETS/DATASET/dictionary_hinglish_transliteration.json', 'r')
    hinglish_dictionary=json.load(jsonfile_transliteration)

    print "check 3"

    tweet_data=data['tweets']
    tweet_transfer_learning_data=trans_data['tweets']

    tweet_return_data = []
    tweet_return_transfer_learning_data=[]
    word_list=[]
    word_list_transfer_learning=[]
    count_Hinglish=0
    count_English=0
    count_total=0
    for i  in range(len(tweet_data)):
        tweet=tweet_data[i]
        emb = 0
        words, temp_pv= glove_tokenize(tweet['text'].lower())
        
        word_list.append(words)

        for w in words:
            if w in word2vec_model:  
                emb+=1
        if emb:   
            tweet_return_data.append(tweet)

    print "check 4"

    for i  in range(len(tweet_transfer_learning_data)):
        tweet=tweet_transfer_learning_data[i]
        emb = 0
        words,count_Hinglish,count_English,count_total,temp_pv= glove_tokenize_transfer_learning(tweet['text'].lower(),hinglish_dictionary,count_Hinglish,count_English,count_total)
        
        for w in words:
            if w in word2vec_model: 
                emb+=1
        if emb:
            tweet_return_transfer_learning_data.append(tweet)
            word_list_transfer_learning.append(words)


    print "check 5"

    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'
    X=[]
    Y=[]
    X_transfer_learning=[]
    Y_transfer_learning=[]

    for i in range(len(word_list)):
        seq=[]
        for word in word_list[i]:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        Y.append(int(tweet_data[i]['label']))

    print "check 6"
    print(len(word_list_transfer_learning))

    for i in range(len(word_list_transfer_learning)):
        seq=[]
        for word in word_list_transfer_learning[i]:
            seq.append(vocab.get(word, vocab['UNK']))
        X_transfer_learning.append(seq)
        # print(tweet_return_transfer_learning_data)

    print "check 7"

    for i in range(len(tweet_return_transfer_learning_data)):
        Y_transfer_learning.append(int(tweet_return_transfer_learning_data[i]['label']))


    MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
    MAX_SEQUENCE_LENGTH_TRANSFER_LEARNING=max(map(lambda x:len(x), X_transfer_learning))
    MAX_SEQUENCE_LENGTH_TOTAL=max(MAX_SEQUENCE_LENGTH,MAX_SEQUENCE_LENGTH_TRANSFER_LEARNING)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH_TOTAL)
    X_transfer_learning=pad_sequences(X_transfer_learning, maxlen=MAX_SEQUENCE_LENGTH_TOTAL)
    Y = np.array(Y)
    Y_transfer_learning=np.array(Y_transfer_learning)

    X_test = X_transfer_learning[:1]
    Y_test = Y_transfer_learning[:1]


    ################ MIMCT MODEL #######################################

    model = load_model('demo_model1.h5')
    model.summary()
    Y_test = np_utils.to_categorical(Y_test, num_classes=3)
    scores = model.evaluate(X_test,Y_test)
    # print(scores[1]*100)
    y_pred=model.predict(X_test)
    y_pred=y_pred.argmax(axis=-1)
    Y_test=Y_test.argmax(axis=-1)

    if y_pred[0] == 0:
        print("Benign")

    elif y_pred[0] == 1:
        print("Hate Inducing")

    elif y_pred[0] == 2:
        print("Abusive")



# print(f1_score(Y_test, y_pred, average='weighted'))
# print(precision_score(Y_test, y_pred, average='weighted'))
# print(recall_score(Y_test, y_pred, average='weighted'))

# Tweet Example 

# @saud5683 @Mutayyab420 @shivang598 @Ranask35 Modi ne desh ko badal diya https://t.co/wZmXZPIMT! - Benign
# @TheKaranPatel @BeingSalmanKhan Shameless teri maa sale kinnar patel harami ki aulad - Abusive
# hindu musalman ki ladai karva denge. - Hate Inducing
# humara BTP acha laga na :) !!! - Benign 


# page
# - scraping
# - graph
# - ss