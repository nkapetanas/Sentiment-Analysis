import string

# from tensorflow import keras
# from keras import Sequential
# from keras.layers.embeddings import Embedding
# from keras.layers import Flatten, Dense
# from keras.preprocessing import sequence
# from numpy import array
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import nltk

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVR

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/Sentiment-Analysis/dataset/train.csv"
DATASET_PATH_TEST = "C:/Users/Delta/PycharmProjects/Sentiment-Analysis/dataset/test_without_labels.csv"

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
stop = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def read_dataset(dataset):
    df = pd.read_csv(dataset, encoding='utf-8')
    return df


def remove_punctuation(text):
    no_punct = "".join([word for word in text if word not in string.punctuation])
    return no_punct


def get_stemmed_text(corpus):
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]


def get_lemmatized_text(corpus):
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]


def clean_data(dataframe):
    dataframe['Content'] = dataframe['Content'].str.lower()
    dataframe['Content'] = dataframe['Content'].str.replace('[^\w\s]', '')
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([item for item in x.split() if item not in stop]))
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))


def calculate_metrics(y_actual, y_predicted):
    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(y_actual, y_predicted)
    recall = recall_score(y_actual, y_predicted)
    f1 = f1_score(y_actual, y_predicted)

    return accuracy, precision, recall, f1


train_data = read_dataset(DATASET_PATH_TRAIN)
test_data = read_dataset(DATASET_PATH_TEST)

clean_data(train_data)
clean_data(test_data)

x_train_data = train_data["Content"]
y_train_data = train_data["Label"]
classes = np.unique(train_data['Label'])

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

fold = 0
for train_index, test_index in kfold.split(x_train_data):
    fold += 1
    print("Fold: %s" % fold)

    x_train_k, x_test_k = x_train_data.iloc[train_index], x_train_data.iloc[test_index]
    y_train_k, y_test_k = y_train_data.iloc[train_index], y_train_data.iloc[test_index]

    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))
    X_train = ngram_vectorizer.fit_transform(x_train_k)
    X_test = ngram_vectorizer.transform(x_test_k)

    sgd_classifier = SGDClassifier(max_iter=1000, loss='hinge')
    sgd_classifier.fit(X_train, y_train_k)
    predictedValues = sgd_classifier.predict(X_test)

    print("Accuracy SGDClassifier: %s"
          % (accuracy_score(y_test_k, predictedValues)))
    accuracy, precision, recall, f1 = metrics = calculate_metrics(y_test_k, predictedValues)
    print("accuracy:" + str(accuracy))
    print("precision:" + str(precision))
    print("recall:" + str(recall))
    print("f1:" + str(f1))
