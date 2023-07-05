'''
Autheur: Eric Pfleiderer (20048976)
Remise: 04/30/19
Cours: PHY3075-Numerical modelling - Modélisation numérique en physique
'''

import pandas as pd
from sklearn import preprocessing


def load_data_sets(split=1750):
    '''
    :param split: index for end of training set and beginning of testing set
    :return: parameters and labels for training set, parameters and labels for testing set
    '''
    x_train = pd.read_csv('data/labeled.csv', sep='\s+').iloc[:split, :-1].values
    y_train = pd.read_csv('data/labeled.csv', sep='\s+').iloc[:split, -1:].values

    x_test = pd.read_csv('data/labeled.csv', sep='\s+').iloc[split:, :-1].values
    y_test = pd.read_csv('data/labeled.csv', sep='\s+').iloc[split:, -1:].values

    # Normalizing data (subtract mean, divide by the std err)
    x_train, x_test = preprocessing.scale(x_train), preprocessing.scale(x_test)

    return x_train, y_train, x_test, y_test
