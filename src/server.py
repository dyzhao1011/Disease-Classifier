from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS

# NLP packages
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tag import pos_tag

# ML packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
stopwords = set(stopwords.words('english'))
punctuations = string.punctuation
stopwords.add('\'m')
stopwords.add('\'ve')

def remove_stopwords(sen):
    return [word.lower() for word in sen if word.lower() not in stopwords]

wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    return [wn.lemmatize(word, pos='v') for word in text]

def remove_punct(tokenized):
    return [word for word in tokenized if word not in punctuations]

def remove_adverbs(tokenized):
    tagged = pos_tag(tokenized)
    filtered = [word for word, tag in tagged if tag != 'RB']
    return filtered

def combine(tokenized):
    return ' '.join(word for word in tokenized)

diseases = pd.DataFrame(pd.read_csv("diseases.csv"))
diseases.pop('Unnamed: 0')
diseases= diseases.rename(columns={'label': 'Disease', 'text': 'Description'})

# Clean Data
diseases['Tokenized'] = diseases['Description'].apply(word_tokenize)
diseases['Removed Stopwords'] = diseases['Tokenized'].apply(remove_stopwords)
diseases['Lemmatized'] = diseases['Removed Stopwords'].apply(lemmatizer)
diseases['Removed Punct'] = diseases['Lemmatized'].apply(remove_punct)
diseases['Removed Adverbs'] = diseases['Removed Punct'].apply(remove_adverbs)
diseases['Symptoms'] = diseases['Removed Adverbs'].apply(combine)

X = diseases['Symptoms']
Y = diseases['Disease']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, stratify = Y)
tfidf_vectorizer = TfidfVectorizer()
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)

svm = SVC(C = 2.0, gamma = 1.0)
svm.fit(X_train_vectorized, Y_train)



app = Flask(__name__)
CORS(app)


@app.route('/extract', methods=['POST'])
def extract():
    data = request.get_json()
    tokenized_data = word_tokenize(data)
    removed_stopwords_data = remove_stopwords(tokenized_data)
    lemmatized_data = lemmatizer(removed_stopwords_data)
    removed_punct_data = remove_punct(lemmatized_data)
    print(removed_punct_data)
    return jsonify({'key_words': removed_punct_data})

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.get_json()

    user_input_vectorized = tfidf_vectorizer.transform([user_input])
    pred_disease = svm.predict(user_input_vectorized)
    print(pred_disease[0])
    return jsonify({'disease': pred_disease[0]})

if __name__ == '__main__':
    app.run(debug = True)