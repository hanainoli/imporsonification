from flask import Flask, request, jsonify
from pydub import AudioSegment
import tempfile
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model


app = Flask(__name__)

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    audio_file = request.files['file']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'})

    if audio_file:
        try:
            # Save the audio file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio_path = temp_audio.name
                audio_file.save(temp_audio_path)
                # initializer = Orthogonal.get(initializer_config)
                # Process the audio file (Example: Getting duration)
                model = load_model('trained_model.h5')
                # audio = AudioSegment.from_file(temp_audio_path)
                response_data = analyze_speech(audio_file, model)

                # You can perform any other processing here...

            # Prepare JSON response
            #response = {
             #   respon
            #}
            
            return jsonify(response_data)

        except Exception as e:
            return jsonify({'error': str(e)})
        finally:
            # Delete temporary file
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    return jsonify({'error': 'Unknown error'})
# Function to extract features from audio files (MFCCs)
def extract_features(audio_file, max_pad_len=174):
    audio_data, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

# Function to analyze emotional tone
def analyze_emotional_tone(audio_file):
    # Load audio data
    audio_data, sample_rate = librosa.load(audio_file, sr=None)
    
    # Compute fundamental frequency (F0) using librosa
    pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
    mean_f0 = np.mean(pitches[pitches > 0])
    
    # Compute intensity (loudness) using root mean square (RMS) energy
    rms_energy = np.sqrt(np.mean(audio_data**2))
    
    # Determine emotional tone based on F0 and intensity
    if mean_f0 > 200 and rms_energy > 0.1:
        emotional_tone = "happy"
    elif mean_f0 < 100 and rms_energy > 0.05:
        emotional_tone = "angry"
    else:
        emotional_tone = "neutral"
    
    return emotional_tone

# Function to analyze background noise level
def analyze_background_noise(audio_file):
    # Load audio data
    audio_data, _ = librosa.load(audio_file, sr=None)
    
    # Compute energy of the audio signal
    signal_energy = np.sum(np.square(audio_data))
    
    # Assume there's a 1-second silent period at the beginning
    silence_duration = 1  # in seconds
    sampling_rate = len(audio_data) / librosa.get_duration(y=audio_data)
    
    # Estimate background noise level based on signal energy
    # We consider the energy of the silent period as noise
    # Assume the background noise is constant throughout the audio
    noise_energy = signal_energy / (len(audio_data) / sampling_rate - silence_duration)
    rms_energy = noise_energy / signal_energy
    
    # Threshold for background noise detection
    noise_threshold = 0.01
    
    # Determine if background noise is present based on RMS energy
    if rms_energy > noise_threshold:
        background_noise_level = "low"
    else:
        background_noise_level = "high"
    
    return background_noise_level

# Function to analyze speech patterns and detect anomalies
def analyze_speech(audio_file, model):
    # Load audio data
    audio_data, _ = librosa.load(audio_file, sr=None)
    
    # Extract features from audio file
    features = extract_features(audio_file)
    
    # Predict anomaly score
    anomaly_score = model.predict(np.array([features]))
    
    # Determine voice type based on anomaly score
    voiceType = "AI" if anomaly_score > 0.5 else "Human"
    
    detectedVoice = True if(voiceType == "Human") else False
    
    # Analyze emotional tone
    emotional_tone = analyze_emotional_tone(audio_file)
    
    # Analyze background noise level
    background_noise_level = analyze_background_noise(audio_file)
    
    # Define confidence score (probability of being human and AI)
    confidence_score = {"AIProbability": anomaly_score[0][0] * 100, "HumanProbability": (1 - anomaly_score[0][0]) * 100}

    return {
        "analysis": {
            "detectedVoice": detectedVoice,
            "voiceType": voiceType,
            "confidenceScore": confidence_score,
            "additionalInfo": {
                "emotionalTone": emotional_tone,
                "backgroundNoiseLevel": background_noise_level
            }
        }
    }

if __name__ == '__main__':
    app.run(debug=True)
