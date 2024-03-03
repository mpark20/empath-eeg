import tensorflow as tf
import keras
#from .preprocessing.dataStreaming import main
#i think we need to add a send/read data function to this dataStreaming file and call it here

#model = tf.keras.models.load_model('classifier.keras')

def process_prompt(role, text, label):
    if role == "assistant":
        return text
    return ''.join([r"[[[", label, r"]]] ", text])

def predict_emotion():
    data = read()
    label = ""
    if data is not None:
        #label = model.predict(data)
        label = "1"
    else:
        label = "0"
    return label

def read():
    #how to read in data from eeg/dataStreaming file?
    print("placeholder")
    return None