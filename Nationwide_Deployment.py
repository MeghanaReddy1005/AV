import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import nltk
import os
import warnings
warnings.filterwarnings('ignore')

import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation
from autocorrect import spell

snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()

from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
stop = set(stopwords.words('english'))
from flask import Flask, jsonify,request
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import nltk
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer 
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
stop_words = stopwords.words('english')


def int_processing(data):
    data.dropna(inplace = True)
    data['Ratings'] = data['Ratings'].astype(int)
    return data

def autospell(text):
    """
    correct the spelling of the word.
    """
    spells = [spell(w) for w in (nltk.word_tokenize(text))]
    return " ".join(spells)

def to_lower(text):
    """
    :param text:
    :return:
        Converted text to lower case as in, converting "Hello" to "hello" or "HELLO" to "hello".
    """
    return text.lower()

def remove_numbers(text):
    """
    take string input and return a clean text without numbers.
    Use regex to discard the numbers.
    """
    output = ''.join(c for c in text if not c.isdigit())
    return output

def remove_punct(text):
    """
    take string input and clean string without punctuations.
    use regex to remove the punctuations.
    """
    return ''.join(c for c in text if c not in punctuation)

def remove_Tags(text):
    """
    take string input and clean string without tags.
    use regex to remove the html tags.
    """
    cleaned_text = re.sub('<[^<]+?>', '', text)
    return cleaned_text

def sentence_tokenize(text):
    """
    take string input and return list of sentences.
    use nltk.sent_tokenize() to split the sentences.
    """
    sent_list = []
    for w in nltk.sent_tokenize(text):
        sent_list.append(w)
    return sent_list

def word_tokenize(text):
    """
    :param text:
    :return: list of words
    """
    return [w for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]

def remove_stopwords(sentence):
    """
    removes all the stop words like "is,the,a, etc."
    """
    stop_words = stopwords.words('english')
    return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

def stem(text):
    """
    :param word_tokens:
    :return: list of words
    """
    stemmed_word = [snowball_stemmer.stem(word) for sent in nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
    return " ".join(stemmed_word)

def lemmatize(text):
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for sent in nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
    return " ".join(lemmatized_word)



def preprocess(message):
    message=message.lower()
    sentence_tokens = sentence_tokenize(message)
    word_list = []
    for each_sent in sentence_tokens:
        lemmatizzed_sent = lemma.lemmatize(each_sent)
        clean_text = remove_numbers(lemmatizzed_sent)
        clean_text = remove_punct(clean_text)
        clean_text = remove_Tags(clean_text)
        clean_text = remove_stopwords(clean_text)
        word_tokens = word_tokenize(clean_text)
    words = word_tokenize(message)
    words_sans_stop=[]
    for word in words :
        if word in stop:continue
        words_sans_stop.append(word)
    return [lemma.lemmatize(word) for word in words_sans_stop]



app = Flask(__name__)

@app.route('/',methods=['POST'])
def home():
    return 'Hello World'

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.get_json(silent=True)
    message=json_.get('message')
    mydf = pd.DataFrame({'message':message})
    print(mydf)
    prediction = clf.predict(mydf['message'])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    clf = joblib.load('my_model_pipeline_Nationwide.pkl')
    app.run(port=5000)