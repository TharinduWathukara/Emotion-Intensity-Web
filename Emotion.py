
print("Importing packages")
import emoji
from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
from nltk.tokenize import word_tokenize
from sklearn.base import TransformerMixin



from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam



import os
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Input, Dense
from sklearn.metrics import jaccard_similarity_score


from spellchecker import SpellChecker
from redditscore.tokenizer import CrazyTokenizer



print("Importing datasets")
data_all = pd.read_csv('data_all_sample_1.csv')

print("Defining preprocessing functions")
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def replaceSmileys(sentence):
    SMILEYS = {
        ":-)":"smile",
        ":-]":"smile",
        ":-3":"smile",
        ":->":"smile",
        "8-)":"smile",
        ":-}":"smile",
        ":)":"smile",
        ":]":"smile",
        ":3":"smile",
        ":>":"smile",
        "8)":"smile",
        ":}":"smile",
        ":o)":"smile",
        ":c)":"smile",
        ":^)":"smile",
        "=]":"smile",
        "=)":"smile",
        ":-))":"smile",
        ":-D":"smile",
        "8-D":"smile",
        "x-D":"smile",
        "X-D":"smile",
        ":D":"smile",
        "8D":"smile",
        "xD":"smile",
        "XD":"smile",
        ":-(":"sad",
        ":-c":"sad",
        ":-<":"sad",
        ":-[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'-('":"sad",
        ":'('":"sad",
        ":-P":"playful",
        "X-P":"playful",
        "x-p":"playful",
        ":-p":"playful",
        ":-b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":b":"playful",
        "<3":"love"
        }
    words = sentence.split()
    reformed = [SMILEYS[word] if word in SMILEYS else word for word in words]
    sentence = " ".join(reformed)
    return sentence

def replaceEmojis(sentence):
    sentence = emoji.demojize(sentence)
    sentence = sentence.replace(":"," ")
    sentence = ' '.join(sentence.split())
    return sentence

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def replaceUnderScore(sentence): #function to clean the word of any punctuation or special characters
    modified = ""
    for word in sentence.split():
        word = word.replace("_"," ")
        modified += word
        modified += " "
    modified = modified.strip()
    return modified

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

def spellCorrect(sentence):
    spell = SpellChecker()
    words = sentence.split()
    misspelled = spell.unknown(words)
    for word in misspelled:
        sentence = sentence.replace(word, spell.correction(word))
    return sentence

def splitHashtags(sentence):
    tokenizer = CrazyTokenizer(hashtags='split')
    sentence = tokenizer.tokenize(sentence)
    return ' '.join(sentence)


sentiment_emotion_lex_file_path ="NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
def get_affect_presence_list(emotion):
    count=-1
    word_list = list()
    alist = []
    with open(sentiment_emotion_lex_file_path) as sentiment_emotion_lex_file:
        for line in sentiment_emotion_lex_file:
            count=count+1
            line=line.replace(" ","\t")
            word_array = line.replace("\n", "").split("\t")
            try:
                if (word_array[1] == emotion and word_array[2] == '1'):
                    word_list.append(word_array[0])
            except:
                print(line)
                print(count)
                
    return word_list

sentiment_emo_lex_word_list_anger = get_affect_presence_list('anger')
sentiment_emo_lex_word_list_fear = get_affect_presence_list('fear')
sentiment_emo_lex_word_list_joy = get_affect_presence_list('joy')
sentiment_emo_lex_word_list_sadness = get_affect_presence_list('sadness')

negs=['no','not','neither','nor']

def negationHandling(Tweet):
    sentence = Tweet.split(' ')
    for i in range(len(sentence)):
        if sentence[i] in negs:
            if(i<len(sentence)-1):
                if sentence[i+1] in sentiment_emo_lex_word_list_joy:
                    Tweet = Tweet.replace(sentence[i]+' '+sentence[i+1], "unsatisfied",1)
                if sentence[i+1] in sentiment_emo_lex_word_list_sadness:
                    Tweet = Tweet.replace(sentence[i]+' '+sentence[i+1], "satisfied",1)
                if sentence[i+1] in sentiment_emo_lex_word_list_fear:
                    Tweet = Tweet.replace(sentence[i]+' '+sentence[i+1], "calmness",1)
                if sentence[i+1] in sentiment_emo_lex_word_list_anger:
                    Tweet = Tweet.replace(sentence[i]+' '+sentence[i+1], "peace",1)
    return Tweet


def pre_process_data(data):
    
    data['Tweet'] = data['Tweet'].apply(replaceSmileys)
    data['Tweet'] = data['Tweet'].apply(clean_text)
    data['Tweet'] = data['Tweet'].str.lower()
    data['Tweet'] = data['Tweet'].apply(cleanHtml)
    data['Tweet'] = data['Tweet'].apply(negationHandling)
    data['Tweet'] = data['Tweet'].apply(replaceEmojis)
    data['Tweet'] = data['Tweet'].apply(replaceUnderScore)
#     data['Tweet'] = data['Tweet'].apply(splitHashtags)
    data['Tweet'] = data['Tweet'].apply(cleanPunc)
    data['Tweet'] = data['Tweet'].apply(keepAlpha)
#     data['Tweet'] = data['Tweet'].apply(spellCorrect)
    return data

# Preprocess dataset
data_all = pre_process_data(data_all)

stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

# Remove stop words from dataset
data_all['Tweet'] = data_all['Tweet'].apply(removeStopWords)

categories = ['anger',  'disgust', 'fear','happiness','sadness','surprise']


# print("Importing wordembedding - glove")
# def _load_words():
#     E = {}
#     vocab = []
#     with open('./../glove.840B.300d.txt', 'r', encoding="utf8") as file:
#         for i, line in enumerate(file):
#             l = line.split(' ')
#             if l[0].isalpha():
#                 v = [float(i) for i in l[1:]]
#                 E[l[0]] = np.array(v)
#                 vocab.append(l[0])
#     return np.array(vocab), E  


# V,E=_load_words()


# def _get_word(v,E,C):
#     for i, emb in enumerate(E):
#         if np.array_equal(emb, v):
#             return V[i]
#     return None


print("Tokenizing")
tokenizer = Tokenizer()

tokenizer.fit_on_texts(data_all['Tweet'])


# TODO
max_length = 40

print('load models')
from keras.models import load_model
model = load_model('model-lstm-multilabel.h5')
anger_model = load_model('model-lstm-intensity-anger.h5')
fear_model = load_model('model-lstm-intensity-fear.h5')
joy_model = load_model('model-lstm-intensity-joy.h5')
sadness_model = load_model('model-lstm-intensity-sadness.h5')



anger = 0.5
anticipation = 0.3
disgust = 0.3
fear = 0.5
joy = 0.5
love = 0.3
optimism = 0.3
pessimism = 0.3
sadness = 0.5
surprise = 0.2
trust = 0.3

def PredictMultiEmotions(text):
    text = replaceSmileys(text)
    text = splitHashtags(text)
    text = clean_text(text)
    text = text.lower()
    text = cleanHtml(text)
    text = negationHandling(text)
    text = replaceEmojis(text)
    text = replaceUnderScore(text)
    text = cleanPunc(text)
    text = keepAlpha(text)
    #text = spellCorrect(text)
    text = removeStopWords(text)

    blist = []
    blist.append(text)
    
    encoded_text = tokenizer.texts_to_sequences(blist)
    text_encoded_padded = pad_sequences(encoded_text, maxlen=max_length, padding='post')
    predictions_for_text = model.predict(text_encoded_padded)
    
    emotions=dict()
    # for x in predictions_for_text:
    #     for label_val in x:
    #         emotions.append(label_val)

    for x in predictions_for_text:
        if x[0]>=anger:
            emotions['anger'] = 1
        else:
            emotions['anger'] = 0
        if x[3]>=fear:
            emotions['fear'] = 1
        else:
            emotions['fear'] = 0
        if x[4]>=joy:
            emotions['joy'] = 1
        else:
            emotions['joy'] = 0
        if x[8]>=sadness:
            emotions['sadness'] = 1
        else:
            emotions['sadness'] = 0

    return emotions


def GetIntensity(text,emotions):
    text = replaceSmileys(text)
    text = splitHashtags(text)
    text = clean_text(text)
    text = text.lower()
    text = cleanHtml(text)
    text = negationHandling(text)
    text = replaceEmojis(text)
    text = replaceUnderScore(text)
    text = cleanPunc(text)
    text = keepAlpha(text)
    #text = spellCorrect(text)
    text = removeStopWords(text)

    blist = []
    blist.append(text)
    
    encoded_text = tokenizer.texts_to_sequences(blist)
    text_encoded_padded = pad_sequences(encoded_text, maxlen=max_length, padding='post')
    
    if(emotions['anger'] == 1):
        anger = anger_model.predict(text_encoded_padded)[0][0]
        anger = float("{0:.2f}".format(anger*100))
    else:
        anger = 0

    if(emotions['fear'] == 1):
        fear = fear_model.predict(text_encoded_padded)[0][0]
        fear = float("{0:.2f}".format(fear*100))
    else:
        fear = 0

    if(emotions['joy'] == 1):
        joy = joy_model.predict(text_encoded_padded)[0][0]
        joy = float("{0:.2f}".format(joy*100))
    else:
        joy = 0

    if(emotions['sadness'] == 1):
        sadness = sadness_model.predict(text_encoded_padded)[0][0]
        sadness = float("{0:.2f}".format(sadness*100))
    else:
        sadness = 0

    output = []
    output.append(anger)
    output.append(fear)
    output.append(joy)
    output.append(sadness)

    return output




# example = ["i like it very"]
# example_encoded = tokenizer.texts_to_sequences(example)
# example_encoded_padded = pad_sequences(example_encoded, maxlen=max_length, padding='post')
# predictions_for_example = model.predict(example_encoded_padded)
# predictions_for_example


example = ["I love to see the latest movie"]
example_encoded = tokenizer.texts_to_sequences(example)
example_encoded_padded = pad_sequences(example_encoded, maxlen=max_length, padding='post')
predictions_for_example = model.predict(example_encoded_padded)
predictions_for_example_anger = anger_model.predict(example_encoded_padded)
predictions_for_example_fear = fear_model.predict(example_encoded_padded)
predictions_for_example_joy = joy_model.predict(example_encoded_padded)
predictions_for_example_sad = sadness_model.predict(example_encoded_padded)
print(predictions_for_example)
print(predictions_for_example_anger)
print(predictions_for_example_fear)
print(predictions_for_example_joy)
print(predictions_for_example_sad)


print('Emotion model complete!')
