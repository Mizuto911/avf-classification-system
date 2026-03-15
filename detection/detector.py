# detection/detector.py
# -*- coding: utf-8 -*-

import joblib
import numpy as np
import librosa

class StenosisDetector:
    THRESHOLD = 90.0  # Classification threshold
    
    def __init__(self, model_path='stenosis_model.pkl', scaler_path='scaler.pkl', sr=22050):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.sample_rate = sr
        self.segment_length = 3
        self.hop_length = 2
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def create_segments(self, audio):
        """Split audio into overlapping segments"""
        seg_samples = int(self.segment_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        segments = []
        for start in range(0, len(audio) - seg_samples, hop_samples):
            segment = audio[start:start + seg_samples]
            segments.append(segment)
        
        return segments
    
    def extract_features(self, audio_segment):
        """Extract acoustic features from audio segment"""
        try:
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sample_rate))
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=self.sample_rate)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
            
            # RMS energy
            rms = np.mean(librosa.feature.rms(y=audio_segment))
            
            # Combine all features
            features = np.concatenate([
                mfccs_mean,
                mfccs_std,
                [spectral_centroid],
                [spectral_rolloff],
                [spectral_bandwidth],
                spectral_contrast_mean,
                [zcr],
                [rms]
            ])
            
            return features
        
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def extract_audio_metrics(self, audio_segment):
        """Extract audio metrics for display"""
        try:
            # RMS energy
            rms = np.sqrt(np.mean(audio_segment**2))
            
            # Peak amplitude
            peak = np.max(np.abs(audio_segment))
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sample_rate))
            
            # Dominant frequency via FFT
            fft = np.fft.rfft(audio_segment)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_segment), 1/self.sample_rate)
            dominant_freq_idx = np.argmax(magnitude)
            dominant_freq = freqs[dominant_freq_idx]
            
            return {
                'rms': rms,
                'peak': peak,
                'zcr': zcr,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'dominant_freq': dominant_freq
            }
        
        except Exception as e:
            print(f"Error extracting metrics: {e}")
            return None
    
    def predict_segment(self, audio_segment):
        """Predict stenosis probability from audio segment"""
        if self.model is None or self.scaler is None:
            return None
        
        features = self.extract_features(audio_segment)
        if features is None:
            return None
            
        features_scaled = self.scaler.transform([features])
        prob = self.model.predict_proba(features_scaled)[0, 1]
        
        return prob
    
    def calculate_confidence(self, normal_prob):
        """Calculate confidence based on distance from threshold"""
        actual = normal_prob * 100
        if actual >= self.THRESHOLD:
            return 100 - (100 - actual)
        else:
            return 100 - (self.THRESHOLD - actual)