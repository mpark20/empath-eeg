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
    Returns the predicted emotion (int) given a row of data
    0=negative, 1=neutral, 2=positive

    data: 1d ndarray of features
    '''    
    # TODO : preprocess into binned data
    input = data.reshape(-1, 1) # reshape into 2d array
    with open('models/classifierV0', 'rb') as f:
        loaded_rf = pickle.load(f)
        pred = loaded_rf.predict(input)  
    return pred

def read():
    #how to read in data from eeg/dataStreaming file?
    print("placeholder")
    return None