import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
import nltk
from nltk.stem.snowball import SnowballStemmer
import regex as re
from nltk.tokenize import sent_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
from nltk.corpus import stopwords

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = stopwords.words('english')

# datasets
df_fake = pd.read_csv('C:\\Users\\CE\\Downloads\\archive (1)\\wrong.csv')
df_true = pd.read_csv('C:\\Users\\CE\\Downloads\\archive (1)\\correct.csv')

# label them seperately
df_true['status'] = 1
df_fake['status'] = 0

# merge and remove unnecessary columns
df = pd.concat([df_true, df_fake])
df.drop(['subject', 'text', 'date'], axis=1, inplace=True)

# let's blend the smoothie
random_indexes = np.random.randint(0, len(df), len(df))
df = df.iloc[random_indexes].reset_index(drop=True)

# text analysis
pd.set_option('display.max_colwidth', 500)
random = np.random.randint(0, len(df), 20)
df.iloc[random]

# Null values
df.isnull().sum()


# longest sentence length
def longest_sentence_length(text):
    return len(text.split())


df['maximum_length'] = df['title'].apply(lambda x: longest_sentence_length(x))
print('longest sentence having length -')
max_length = max(df['maximum_length'].values)
print(max_length)

# Text cleaning
text_cleaning = "\b0\S*|\b[^A-Za-z0-9]+"


def preprocess_filter(text, stem=False):
    text = re.sub(text_cleaning, " ", str(text.lower()).strip())
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                stemmer = SnowballStemmer(language='english')
                token = stemmer.stem(token)
            tokens.append(token)
    return " ".join(tokens)


# Word embedding with pre padding
def one_hot_encoded(text, vocab_size=5000, max_length=40):
    hot_encoded = one_hot(text, vocab_size)
    return hot_encoded


# word embedding pipeline
def word_embedding(text):
    preprocessed_text = preprocess_filter(text)
    return one_hot_encoded(preprocessed_text)


# Creating NN Model
embedded_features = 40
model = Sequential()
model.add(Embedding(5000, embedded_features, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# One hot encoded title
one_hot_encoded_title = df['title'].apply(lambda x: word_embedding(x)).values

# padding to make the size equal of the sequences
padded_encoded_title = pad_sequences(one_hot_encoded_title, maxlen=max_length, padding='pre')

# Splitting
X = padded_encoded_title
y = df['status'].values
y = np.array(y)

# shapes
print(X.shape)
print(y.shape)

# shape and size
print('X shape {}'.format(X.shape))
print('y shape {}'.format(y.shape))

# Splitting into training, testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Shape and size of train and test dataset
print('X train shape {}'.format(X_train.shape))
print('X test shape {}'.format(X_test.shape))
print('y train shape {}'.format(y_train.shape))
print('y test shape {}'.format(y_test.shape))

# Model training
# training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=64)


# evolution
# setting threshold value
def best_threshold_value(thresholds: list, X_test):
    accuracies = []
    for thresh in thresholds:
        ypred = model.predict(X_test)
        ypred = np.where(ypred > thresh, 1, 0)
        accuracies.append(accuracy_score(y_test, ypred))
    return pd.DataFrame({
        'Threshold': thresholds,
        'Accuracy': accuracies
    })


best_threshold_value([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], X_test)

# Predictino value at threshold 0.4
y_pred = model.predict(X_test)
y_pred = np.where(y_pred > 0.4, 1, 0)

# Confusion matrix
print('Confusion matrix')
print(confusion_matrix(y_pred, y_test))
print('----------------')
print('Classification report')
print(classification_report(y_pred, y_test))


# input generator
def prediction_input_processing(text):
    encoded = word_embedding(text)
    padded_encoded_title = pad_sequences([encoded], maxlen=max_length, padding='pre')
    output = model.predict(padded_encoded_title)
    output = np.where(0.4 > output, 1, 0)
    if output[0][0] == 1:
        return 'Yes this News is fake'
    return 'No, It is not fake'


# predictions
prediction_input_processing('Rubio running considers')

news = 'John Oliver Offers Harsh Critique Of Dem Primary, Tells Us Who Should Be Winning (VIDEO)'
prediction_input_processing(news)
