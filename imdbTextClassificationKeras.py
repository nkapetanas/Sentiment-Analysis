import string

import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import StratifiedKFold

DATASET_PATH_TRAIN = "C:/Users/Delta/PycharmProjects/Sentiment-Analysis/dataset/train.csv"
DATASET_PATH_TEST = "C:/Users/Delta/PycharmProjects/Sentiment-Analysis/dataset/test_without_labels.csv"

labels = []
maxlen = 100
training_samples = 20000
validation_samples = 5000
max_words = 10000

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
stop = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def load_dataset(dataset):
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
    dataframe['Content'] = dataframe['Content'].str.replace('<[^<]+?>', '')  # remove HTML tags
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([item for item in x.split() if item not in stop]))
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    dataframe['Content'] = dataframe['Content'].apply(
        lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

def createCSV(prediction, csvName):
    df = pd.DataFrame(prediction, columns=['Predicted'], index=np.arange(0, prediction.size))
    df.index.name = 'Id'
    df.to_csv(csvName)

def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_kfold(x_train, y_train, k):
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(x_train, y_train))
    return folds, x_train, y_train


def get_model():
    # create the model
    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=maxlen))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                  metrics=['accuracy', recall_metric, precision_metric, f1])

    return model


def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1,
                                       mode='min')
    return [mcp_save, reduce_lr_loss]

scores_accuracy = []
scores_precision = []
scores_recall = []
scores_f1 = []

tokenizer = Tokenizer(num_words=max_words)

train_data = load_dataset(DATASET_PATH_TRAIN)
test_data = load_dataset(DATASET_PATH_TEST)

clean_data(train_data)
clean_data(test_data)

x_train_data = train_data["Content"]
y_train_data = np.asarray(train_data["Label"]).astype('float32')

tokenizer.fit_on_texts(x_train_data)
sequences = tokenizer.texts_to_sequences(x_train_data)

word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.array(y_train_data)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_test = data[training_samples:]
y_test = labels[training_samples:]

folds, x_train, y_train = get_kfold(x_train, y_train, 5)

model = get_model()
model.summary()

for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ', j)
    X_train_cv = x_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = x_train[val_idx]
    y_valid_cv = y_train[val_idx]

    name_weights = "final_model_fold" + str(j) + "_weights.h5"
    callbacks = get_callbacks(name_weights=name_weights, patience_lr=10)

    model.fit(X_valid_cv, y_valid_cv, batch_size=32, epochs=5, verbose=2, callbacks=callbacks)
    score = model.evaluate(X_valid_cv, y_valid_cv, verbose=0)

    scores_accuracy.append(score[0])
    scores_recall.append(score[1])
    scores_precision.append(score[2])
    scores_f1.append(score[3])

print("Keras Model metrics")
print("Accuracy:" + str(np.mean(scores_accuracy)))
print("Precision:" + str(np.mean(scores_precision)))
print("Recall:" + str(np.mean(scores_recall)))
print("F1:" + str(np.mean(scores_f1)))

sequences = tokenizer.texts_to_sequences(test_data['Content'])
x_test = pad_sequences(sequences, maxlen=maxlen)
pred = model.predict_classes(x_test)

createCSV(pred, "sentiment_predictions.csv")