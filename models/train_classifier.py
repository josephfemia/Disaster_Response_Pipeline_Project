import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

import pickle


def load_data(database_filepath):
    '''
    INPUT:
        database_filepath - the path where the database file is located
    OUTPUT:
        X - returns a dataframe containing only the messages
        Y - returns a dataframe containing all categories related to the message
        category_names - returns the category titles
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    INPUT:
        text - a string of words
    OUTPUT:
        clean_tokens - a list of words that are considered to have a lot of
        information about the text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        if token not in stopwords.words('english'):
            clean_tok = lemmatizer.lemmatize(token).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    INPUT:
        no input
    OUTPUT:
        pipeline - returns a model used to evaluate the cleaned tokens
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))
    ])

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
        model - the model used to make predictions about the data
        X_test - messages used to test the model's performance
        Y_test - dataframe containing correct category selection relating to
        each message
    OUTPUT:
        prints out a report of the model's performance for every category name
    '''
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)

    for i, col in enumerate(category_names):
        print('Category: ', col)
        print(classification_report(Y_test.iloc[:, i], y_pred.iloc[:, i]))


def save_model(model, model_filepath):
    '''
    INPUT:
        model - the model used to make predictions about the data
        model_filepath - the path where the user wants the model to be saved to
    OUTPUT:
        saves the model located at the specified path
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
