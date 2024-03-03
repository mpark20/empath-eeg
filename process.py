#from .preprocessing.dataStreaming import main
#placeholder for preprocessing + read raw data functions 

import pickle
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def process_prompt(role, text, label):
    '''
    Prepends emotion state to prompt if role is user
    '''
    if role == "assistant":
        return text
    return ''.join([r"[[[", label, r"]]] ", text])

def predict_emotion():
    '''
    Returns the predicted emotion given a row of data
    '''   
    # TODO: read in raw data from other class 
    data = []
    # TODO : preprocess into binned data
    if data == []:
        datas = read_sample()
        data = datas[random.choice(range(datas.shape[0]))]
    input = data.reshape(1, -1)
    with open('../models/classifierV0', 'rb') as f:
        loaded_rf = pickle.load(f)
        pred = loaded_rf.predict(input)  
    return pred

def read_sample():
    data = pd.read_csv('./data/emotions.csv')
    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    nan_count = np.sum(data.isnull(), axis = 0)
    nan_cols = nan_count != 0
    #splits the columns with null values into numerical and categorical dfs
    col_w_nan = data.columns[nan_count != 0]
    num_columns = [col for col in col_w_nan if data[col].dtype == 'int64' or data[col].dtype == 'float64']
    for col in num_columns:
        data = replaceNullNum(data, col)
    data.dropna(inplace = True)
    _, X_test, _, _ = preprocess_data(data, label_mapping)
    return X_test

def replaceNullNum(data, col):
  mean = data[col].mean()
  data[col].fillna(value=mean, inplace=True)
  return data

def preprocess_data(df, label_mapping):
    df = df.copy()

    df['label'] = df['label'].replace(label_mapping)

    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)

    return X_train, X_test, y_train, y_test