from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow.keras import layers

app = Flask(__name__)

# Define the model architecture
def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=5, strides=2, padding='same'),
        
        layers.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=5, strides=2, padding='same'),
        layers.Dropout(0.2),
        
        layers.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=5, strides=2, padding='same'),
        
        layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=5, strides=2, padding='same'),
        layers.Dropout(0.2),
        
        layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(7, activation='softmax')
    ])
    return model

# Load the model and necessary files
model = create_model((None, 1))  # Adjust the input shape as needed
model.load_weights('CNN_model_weights.weights.h5')

with open('encoder2.pickle', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler2.pickle', 'rb') as f:
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    
    # Load the audio
    audio, sample_rate = librosa.load(audio_file, duration=3, offset=0.5)
    
    # Extract features
    features = extract_features(audio, sr=sample_rate)
    
    # Scale the features
    features = scaler.transform(features.reshape(1, -1))
    
    # Reshape for model input
    features = features.reshape(1, -1, 1)
    
    # Make prediction
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    
    # Decode the prediction
    emotion = encoder.inverse_transform([predicted_class])[0]
    
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)