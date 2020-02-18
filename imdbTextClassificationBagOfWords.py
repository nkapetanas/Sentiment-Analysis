import string

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_precision_recall_curve
from sklearn.model_selection import KFold, validation_curve

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/Sentiment-Analysis/dataset/train.csv"
DATASET_PATH_TEST = "C:/Users/Delta/PycharmProjects/Sentiment-Analysis/dataset/test_without_labels.csv"

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def createCSV(prediction, csvName):
    np.savetxt(csvName,
               np.dstack((np.arange(0, prediction.size), prediction))[0], "%d,%d",
               header="Id,Predicted")


def read_dataset(dataset):
    df = pd.read_csv(dataset, encoding='utf-8')
    return df

def clean_data(dataframe):
    dataframe['Content'] = dataframe['Content'].str.lower()
    dataframe['Content'] = dataframe['Content'].str.replace('[^\w\s]', '')
    dataframe['Content'] = dataframe['Content'].str.replace('<[^<]+?>', '')  # remove HTML tags
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([item for item in x.split() if item not in stop]))
    # dataframe['Content'] = dataframe['Content'].apply(
    #     lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))


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

test_data_ = test_data['Content']

scores_svm_accuracy = []
scores_svm_precision = []
scores_svm_recall = []
scores_svm_f1 = []

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3))

kfold = KFold(n_splits=5, random_state=42, shuffle=True)

sgd_classifier = SGDClassifier(max_iter=1000, loss='hinge')
fold = 0

for train_index, test_index in kfold.split(x_train_data):
    fold += 1
    print("Fold: %s" % fold)

    x_train_k, x_test_k = x_train_data.iloc[train_index], x_train_data.iloc[test_index]
    y_train_k, y_test_k = y_train_data.iloc[train_index], y_train_data.iloc[test_index]

    X_train = ngram_vectorizer.fit_transform(x_train_k)
    X_test = ngram_vectorizer.transform(x_test_k)

    sgd_classifier.fit(X_train, y_train_k)
    predictedValues = sgd_classifier.predict(X_test)

    accuracy, precision, recall, f1 = metrics = calculate_metrics(y_test_k, predictedValues)

    scores_svm_accuracy.append(accuracy)
    scores_svm_precision.append(precision)
    scores_svm_recall.append(recall)
    scores_svm_f1.append(f1)

print("SGDClassifier metrics")
print("Accuracy:" + str(np.mean(scores_svm_accuracy)))
print("Precision:" + str(np.mean(scores_svm_precision)))
print("Recall:" + str(np.mean(scores_svm_recall)))
print("F1:" + str(np.mean(scores_svm_f1)))

test_data_ = ngram_vectorizer.transform(test_data_)

predictedValues = sgd_classifier.predict(test_data_)
createCSV(predictedValues, "sentiment_predictions.csv")

# X_train = ngram_vectorizer.fit_transform(x_train_data)
# plot_precision_recall_curve(sgd_classifier, X_train, y_train_data)
# plt.show()
