from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow.keras import layers

dep2 = Flask(__name__)

# Define the model architecture
import tensorflow as tf

from tensorflow.keras.models import Sequential, model_from_json
json_file = open('C:/V/python/speech_emotions_detection/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/V/python/speech_emotions_detection/model_weights.weights.h5")


with open('C:/V/python/speech_emotions_detection/encoder2.pickle', 'rb') as f:
    encoder = pickle.load(f)

with open('C:/V/python/speech_emotions_detection/scaler2.pickle', 'rb') as f:
    scaler = pickle.load(f)

def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten:bool=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    
    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                       ))
    return result




@dep2.route('/')
def index():
    return render_template('index2.html')

@dep2.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    # Load the audio
    d, s_rate= librosa.load(audio_file, duration=2.5, offset=0.6)
    # Extract features
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,2376))
    # Scale the features
    i_result = scaler2.transform(result)
    res = np.expand_dims(i_result, axis=2)
    
    # Make prediction
    emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
    predictions=loaded_model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    print(y_pred[0][0]) 
    
    return render_template('index2.html', prediction_text='Emotion $ {}'.format(y_pred[0][0]))
    # return jsonify({'emotion': emotion})

if __name__ == '__main__':
    dep2.run(debug=False)