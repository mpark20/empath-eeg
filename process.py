# import tensorflow as tf
# import keras
#from .preprocessing.dataStreaming import main
#i think we need to add a send/read data function to this dataStreaming file and call it here

#model = tf.keras.models.load_model('classifier.keras')
import pickle

def process_prompt(role, text, label):
    if role == "assistant":
        return text
    return ''.join([r"[[[", label, r"]]] ", text])

def predict_emotion(data):
    '''
    Returns the predicted emotion given a row of data
    '''    
    # TODO : preprocess into binned data
    with open('../models/classifierV0', 'rb') as f:
        loaded_rf = pickle.load(f)
        pred = loaded_rf.predict(data)  
    return pred

def read():
    #how to read in data from eeg/dataStreaming file?
    print("placeholder")
    return None