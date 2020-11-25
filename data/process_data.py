import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
        messages_filepath - the path to where the messages csv file is located
        categories_filepath - the path to where the categories csv file is located
    OUTPUT:
        returns a dataframe that contains the information from both of the files
        that were passed.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how = 'outer', on = 'id')

    return df


def clean_data(df):
    '''
    INPUT:
        df - dataframe that needs to be cleaned
    OUTPUT:
        returns a cleaned dataframe
    '''
    categories = df['categories'].str.split(';', expand = True)

    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1:])
        categories[column] = pd.to_numeric(categories[column])

    df.drop(['categories'], axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)

    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename):
    '''
    INPUT:
        df - a dataframe that contains all of the information to be input into
        the database
        database_filename - the name of the database
    OUTPUT:
        nothing gets returned but a sql database gets made with the specificed
        name
    '''
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('{}'.format(database_filename), engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
