import time
import numpy as np
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam

from keras.models import Sequential, save_model, load_model

from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Dense, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D


class Keras:
    def __init__(self, df, max_words=5000, batch_size=250, no_epochs=6, validation_split=0.2, verbosity=1, max_len=50, embedding_dims=None):
        self.df = df
        self.max_words = max_words
        self.batch_size = batch_size
        self.no_epochs = no_epochs
        self.validation_split = validation_split
        self.verbosity = verbosity
        self.max_len = max_len
        self.embedding_dims = embedding_dims

class RNN(Keras):
    def create(self):
        print('\nStart creating RNN...')
        start = time.time()

        X = self.df.Review
        y = self.df.Label

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation_split, random_state=42)

        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(X_train)
        self.tokenizer = tokenizer

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        X_train = pad_sequences(X_train, padding='post', maxlen=self.max_len)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.max_len)

        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1

        # create the model
        model = Sequential()
        model.add(Embedding(vocab_size, 32, input_length=self.max_len))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=3e-4)

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['acc'])
        print(model.summary())

        # Fit the model
        history=model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.no_epochs, verbose=self.verbosity, validation_split=self.validation_split)

        # Final evaluation of the model
        self.model = model
        end = time.time()
        print(f'Finished creating RNN in {str(end-start)} seconds')

        score = model.evaluate(X_test, y_test, verbose=1)
        y_prediction = model.predict(X_test)
        y_prediction = np.round_(y_prediction)
        cm = confusion_matrix(y_test, y_prediction)
        # plot_confusion_matrix(cm, ['negative', 'positive'])

        print(model.metrics_names)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        print(f'Confusion Matrix:\n', cm)
        print(f'Classification Report:\n', classification_report(y_test, y_prediction))


        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])

        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train','test'], loc='upper left')
        # plt.show()


class CNN(Keras):
    def create(self):
        print('\nStart creating CNN...')
        start = time.time()

        X = self.df.Review
        y = self.df.Label

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation_split, random_state=42)

        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(X_train)
        self.tokenizer = tokenizer

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        X_train = pad_sequences(X_train, padding='post', maxlen=self.max_len)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.max_len)

        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1

        # create the model
        model = Sequential()
        # embedding layer (vocab_size is the total number of words in data,
        # then the embedding dimensions we specified, then the maximum length of one review)
        model.add(Embedding(vocab_size, self.embedding_dims, input_length=self.max_len))
        model.add(Conv1D(128, kernel_size=4, input_shape=(vocab_size, self.embedding_dims),activation="relu"))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(LSTM(32, recurrent_dropout=0.4))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()


        model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
        model.fit(X_train, y_train, epochs=self.no_epochs,batch_size=self.batch_size,verbose=self.verbosity, validation_split=self.validation_split)

        self.model = model
        end = time.time()
        print(f'Finished creating RNN in {str(end-start)} seconds')

        score = model.evaluate(X_test, y_test, verbose=1)
        y_prediction = model.predict(X_test)
        y_prediction = np.round_(y_prediction)
        cm = confusion_matrix(y_test, y_prediction)
        # plot_confusion_matrix(cm, ['negative', 'positive'])

        print(model.metrics_names)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        print(f'Confusion Matrix:\n', cm)
        print(f'Classification Report:\n', classification_report(y_test, y_prediction))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
