# coding: utf-8
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import librosa
import numpy as np
import joblib
from pathlib import Path
import threading
import warnings
import sounddevice as sd
from datetime import datetime
import queue
import time

# Matplotlib for visualization
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

warnings.filterwarnings('ignore')

class StenosisTester:
    THRESHOLD = 90.0
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AVF Stenosis Detector")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg='#1a1a1a')
        
        # Model parameters
        self.model = None
        self.scaler = None
        self.segment_length = 3
        self.hop_length = 2
        self.sample_rate = 22050
        
        # Recording parameters
        self.mic_device_id = None
        self.mic_sample_rate = 48000
        
        # Auto-detect USB audio device
        self.detect_usb_audio()
        
        # Live detection parameters
        self.is_live_detecting = False
        self.audio_queue = queue.Queue()
        self.live_buffer = []
        self.live_buffer_size = self.mic_sample_rate * 3
        self.live_predictions = []
        self.live_prediction_window = 10
        
        # Audio metrics storage
        self.current_audio_metrics = {
            'rms': 0,
            'peak': 0,
            'zcr': 0,
            'spectral_centroid': 0,
            'spectral_rolloff': 0,
            'spectral_bandwidth': 0,
            'dominant_freq': 0
        }
        
        self.setup_ui()
        self.load_model()
    
    def detect_usb_audio(self):
        """Auto-detect USB audio device"""
        try:
            devices = sd.query_devices()
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    if 'usb' in device_name or 'ab13x' in device_name:
                        self.mic_device_id = i
                        self.mic_sample_rate = int(device['default_samplerate'])
                        self.device_name = device['name']
                        print(f"Found USB audio: {device['name']} (Device {i})")
                        return
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.mic_device_id = i
                    self.mic_sample_rate = int(device['default_samplerate'])
                    self.device_name = device['name']
                    print(f"Using device: {device['name']} (Device {i})")
                    return
            
            self.device_name = "No input device found"
            
        except Exception as e:
            print(f"Error detecting audio device: {e}")
            self.device_name = "Error detecting device"
    
    def setup_ui(self):
        # Compact header bar
        header_frame = tk.Frame(self.window, bg='#0d47a1', height=50)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="AVF Stenosis Detector",
            font=("Arial", 16, "bold"),
            bg='#0d47a1',
            fg='white'
        )
        title_label.pack(side='left', padx=15, pady=10)
        
        # Device info in header
        self.device_label = tk.Label(
            header_frame,
            text=f"{self.device_name} | {self.mic_sample_rate} Hz",
            font=("Arial", 9),
            bg='#0d47a1',
            fg='#bbdefb'
        )
        self.device_label.pack(side='left', padx=15)
        
        # Model status
        self.model_status = tk.Label(
            header_frame,
            text="Model: Loading...",
            font=("Arial", 9),
            bg='#0d47a1',
            fg='#bbdefb'
        )
        self.model_status.pack(side='right', padx=15)
        
        # Close button
        close_btn = tk.Button(
            header_frame,
            text="?",
            command=self.window.quit,
            font=("Arial", 16, "bold"),
            bg='#d32f2f',
            fg='white',
            activebackground='#b71c1c',
            cursor="hand2",
            relief='flat',
            width=3
        )
        close_btn.pack(side='right', padx=10)
        
        # Main container
        main_container = tk.Frame(self.window, bg='#1a1a1a')
        main_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Top section - compact button and result
        top_frame = tk.Frame(main_container, bg='#2c2c2c', relief='solid', borderwidth=1)
        top_frame.pack(fill='x', pady=(0, 10))
        
        # Button and result in one row
        control_frame = tk.Frame(top_frame, bg='#2c2c2c')
        control_frame.pack(fill='x', padx=15, pady=10)
        
        # Start button - compact
        self.start_btn = tk.Button(
            control_frame,
            text="START LISTENING",
            command=self.toggle_listening,
            font=("Arial", 12, "bold"),
            bg='#4caf50',
            fg='white',
            activebackground='#45a049',
            cursor="hand2",
            relief='flat',
            pady=8,
            padx=20
        )
        self.start_btn.pack(side='left', padx=(0, 20))
        
        # Status indicator
        self.status_indicator = tk.Label(
            control_frame,
            text="? Ready",
            font=("Arial", 11, "bold"),
            bg='#2c2c2c',
            fg='#9e9e9e'
        )
        self.status_indicator.pack(side='left', padx=(0, 30))
        
        # Result display - compact
        result_frame = tk.Frame(control_frame, bg='#2c2c2c')
        result_frame.pack(side='left', padx=(0, 20))
        
        tk.Label(
            result_frame,
            text="Result:",
            font=("Arial", 10, "bold"),
            bg='#2c2c2c',
            fg='#bdbdbd'
        ).pack(side='left', padx=(0, 10))
        
        self.result_label = tk.Label(
            result_frame,
            text="READY",
            font=("Arial", 18, "bold"),
            bg='#2c2c2c',
            fg='#9e9e9e'
        )
        self.result_label.pack(side='left')
        
        # Confidence display - compact
        confidence_frame = tk.Frame(control_frame, bg='#2c2c2c')
        confidence_frame.pack(side='left')
        
        tk.Label(
            confidence_frame,
            text="Confidence:",
            font=("Arial", 10, "bold"),
            bg='#2c2c2c',
            fg='#bdbdbd'
        ).pack(side='left', padx=(0, 10))
        
        self.confidence_label = tk.Label(
            confidence_frame,
            text="--",
            font=("Arial", 14, "bold"),
            bg='#2c2c2c',
            fg='#ffffff'
        )
        self.confidence_label.pack(side='left')
        
        # Data area - 3 columns
        data_frame = tk.Frame(main_container, bg='#1a1a1a')
        data_frame.pack(fill='both', expand=True)
        
        # LEFT COLUMN - Analysis Data
        left_column = tk.Frame(data_frame, bg='#2c2c2c', relief='solid', borderwidth=1)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.create_data_section(left_column, "ANALYSIS DATA", [
            ("Duration", "duration"),
            ("Samples", "samples"),
            ("Consistency", "consistency"),
            ("Stenosis Prob", "stenosis_prob"),
            ("Normal Prob", "normal_prob")
        ])
        
        # MIDDLE COLUMN - Audio Metrics
        middle_column = tk.Frame(data_frame, bg='#2c2c2c', relief='solid', borderwidth=1)
        middle_column.pack(side='left', fill='both', expand=True, padx=5)
        
        self.create_data_section(middle_column, "AUDIO METRICS", [
            ("RMS Energy", "rms"),
            ("Peak Amplitude", "peak"),
            ("Zero Cross Rate", "zcr"),
            ("Dominant Freq", "dominant_freq"),
            ("Spectral Centroid", "centroid"),
            ("Spectral Rolloff", "rolloff"),
            ("Spectral Bandwidth", "bandwidth")
        ])
        
        # RIGHT COLUMN - Visualization
        right_column = tk.Frame(data_frame, bg='#2c2c2c', relief='solid', borderwidth=1)
        right_column.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        viz_header = tk.Label(
            right_column,
            text="VISUALIZATION",
            font=("Arial", 11, "bold"),
            bg='#2c2c2c',
            fg='#ffffff',
            pady=8
        )
        viz_header.pack(fill='x')
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 6), facecolor='#2c2c2c')
        
        # Waveform plot
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax1.set_facecolor('#1a1a1a')
        self.ax1.set_title('Waveform', color='white', fontsize=10, pad=5)
        self.ax1.set_ylabel('Amplitude', color='white', fontsize=9)
        self.ax1.tick_params(colors='white', labelsize=8)
        self.line1, = self.ax1.plot([], [], color='#00bcd4', linewidth=1.5)
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True, alpha=0.2, color='white')
        
        # Spectrum plot
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.ax2.set_facecolor('#1a1a1a')
        self.ax2.set_title('Frequency Spectrum (0-2kHz)', color='white', fontsize=10, pad=5)
        self.ax2.set_xlabel('Frequency (Hz)', color='white', fontsize=9)
        self.ax2.set_ylabel('Magnitude', color='white', fontsize=9)
        self.ax2.tick_params(colors='white', labelsize=8)
        self.line2, = self.ax2.plot([], [], color='#ff5722', linewidth=1.5)
        self.ax2.set_xlim(0, 2000)
        self.ax2.grid(True, alpha=0.2, color='white')
        
        self.fig.tight_layout(pad=2.0)
        
        # Embed canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_column)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=(0, 10))
    
    def create_data_section(self, parent, title, metrics):
        """Create a data section with metrics"""
        # Header
        header = tk.Label(
            parent,
            text=title,
            font=("Arial", 11, "bold"),
            bg='#2c2c2c',
            fg='#ffffff',
            pady=8
        )
        header.pack(fill='x')
        
        # Metrics container
        metrics_container = tk.Frame(parent, bg='#2c2c2c')
        metrics_container.pack(fill='both', expand=True, padx=15, pady=10)
        
        for label_text, key in metrics:
            self.create_metric_row(metrics_container, label_text, key)
    
    def create_metric_row(self, parent, label_text, key):
        """Create a metric row with label and value"""
        frame = tk.Frame(parent, bg='#2c2c2c')
        frame.pack(fill='x', pady=4)
        
        # Label
        label = tk.Label(
            frame,
            text=f"{label_text}:",
            font=("Arial", 10, "bold"),
            bg='#2c2c2c',
            fg='#bdbdbd',
            anchor='w'
        )
        label.pack(side='left', fill='x', expand=True)
        
        # Value
        value = tk.Label(
            frame,
            text="--",
            font=("Arial", 11),
            bg='#2c2c2c',
            fg='#4caf50',
            anchor='e',
            width=15
        )
        value.pack(side='right')
        
        # Store reference
        setattr(self, f"{key}_value", value)
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            script_dir = Path(__file__).parent.absolute()
            model_path = script_dir / 'stenosis_model.pkl'
            scaler_path = script_dir / 'scaler.pkl'
            
            self.model = joblib.load(str(model_path))
            self.scaler = joblib.load(str(scaler_path))
            
            self.model_status.config(
                text="Model: Ready ?",
                fg='#81c784'
            )
            
        except FileNotFoundError:
            self.model_status.config(
                text="Model: NOT FOUND ?",
                fg='#e57373'
            )
            messagebox.showerror(
                "Model Not Found",
                "Cannot find model files"
            )
        except Exception as e:
            self.model_status.config(
                text=f"Model: ERROR ?",
                fg='#e57373'
            )
    
    def toggle_listening(self):
        """Start or stop listening"""
        if not self.is_live_detecting:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        """Start live detection"""
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return
        
        if self.mic_device_id is None:
            messagebox.showerror("No Device", "No audio input device detected")
            return
        
        self.is_live_detecting = True
        
        # Clear buffers
        self.live_buffer = []
        self.live_predictions = []
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Update UI
        self.start_btn.config(
            text="STOP LISTENING",
            bg='#f44336',
            activebackground='#d32f2f'
        )
        
        self.status_indicator.config(
            text="? Listening",
            fg='#4caf50'
        )
        
        self.result_label.config(
            text="...",
            fg='#ff9800'
        )
        
        self.confidence_label.config(text="...")
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.mic_sample_rate,
                channels=1,
                device=self.mic_device_id,
                callback=self.audio_callback,
                blocksize=int(self.mic_sample_rate * 0.1),
                dtype='float32'
            )
            self.stream.start()
            
            # Start processing thread
            process_thread = threading.Thread(target=self._process_live_audio)
            process_thread.daemon = True
            process_thread.start()
            
        except Exception as e:
            self.stop_listening()
            messagebox.showerror("Stream Error", f"Failed to start audio:\n\n{str(e)}")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback"""
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def _process_live_audio(self):
        """Process live audio continuously"""
        while self.is_live_detecting:
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
                self.live_buffer.extend(audio_chunk.flatten())
                
                if len(self.live_buffer) > self.live_buffer_size:
                    self.live_buffer = self.live_buffer[-self.live_buffer_size:]
                
                if len(self.live_buffer) >= self.live_buffer_size:
                    # Update visualization
                    self.window.after(0, self.update_visualization, np.array(self.live_buffer))
                    
                    # Resample for analysis
                    audio_segment = np.array(self.live_buffer)
                    audio_resampled = librosa.resample(
                        audio_segment,
                        orig_sr=self.mic_sample_rate,
                        target_sr=self.sample_rate
                    )
                    
                    # Extract metrics
                    self.extract_and_store_metrics(audio_resampled)
                    self.window.after(0, self.update_audio_metrics_display)
                    
                    # Extract features and predict
                    features = self.extract_features(audio_resampled)
                    
                    if features is not None:
                        features_scaled = self.scaler.transform([features])
                        prob = self.model.predict_proba(features_scaled)[0, 1]
                        
                        self.live_predictions.append(prob)
                        
                        if len(self.live_predictions) > self.live_prediction_window:
                            self.live_predictions = self.live_predictions[-self.live_prediction_window:]
                        
                        mean_prob = np.mean(self.live_predictions)
                        self.window.after(0, self._update_results, mean_prob, len(self.live_predictions))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def _update_results(self, prob, num_samples):
        """Update results display"""
        if not self.is_live_detecting:
            return
        
        stenosis_threshold = 1.0 - (self.THRESHOLD / 100.0)
        prediction = "STENOSIS" if prob > stenosis_threshold else "NORMAL"
        
        normal_prob = 1 - prob
        stenosis_prob = prob
        confidence = self.calculate_threshold_confidence(normal_prob)
        
        # Color coding
        if prediction == 'STENOSIS':
            color = "#f44336"
            status_color = "#f44336"
            status_text = "? Stenosis"
        else:
            color = "#4caf50"
            status_color = "#4caf50"
            status_text = "? Normal"
        
        self.result_label.config(text=prediction, fg=color)
        self.confidence_label.config(text=f"{confidence:.1f}%")
        self.status_indicator.config(text=status_text, fg=status_color)
        
        # Update analysis data
        self.samples_value.config(text=f"{num_samples}")
        self.duration_value.config(text="Live")
        self.stenosis_prob_value.config(text=f"{stenosis_prob*100:.1f}%")
        self.normal_prob_value.config(text=f"{normal_prob*100:.1f}%")
        
        if len(self.live_predictions) > 1:
            std_prob = np.std(self.live_predictions)
            consistency = 1 - std_prob
            self.consistency_value.config(text=f"{consistency:.1%}")
    
    def stop_listening(self):
        """Stop live detection"""
        self.is_live_detecting = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.start_btn.config(
            text="START LISTENING",
            bg='#4caf50',
            activebackground='#45a049'
        )
        
        self.status_indicator.config(
            text="? Ready",
            fg='#9e9e9e'
        )
        
        # Keep final results visible
    
    def update_visualization(self, audio_data):
        """Update visualization plots"""
        try:
            # Waveform
            if len(audio_data) > 1000:
                waveform_data = audio_data[-1000:]
            else:
                waveform_data = audio_data
            
            x_wave = np.arange(len(waveform_data))
            self.line1.set_data(x_wave, waveform_data)
            self.ax1.set_xlim(0, len(waveform_data))
            
            # Spectrum
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.mic_sample_rate)
            
            freq_mask = freqs <= 2000
            freqs_display = freqs[freq_mask]
            magnitude_display = magnitude[freq_mask]
            
            self.line2.set_data(freqs_display, magnitude_display)
            max_mag = np.max(magnitude_display)
            self.ax2.set_ylim(0, max_mag * 1.1 if max_mag > 0 else 1)
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def extract_and_store_metrics(self, audio_segment):
        """Extract audio metrics"""
        try:
            rms = np.sqrt(np.mean(audio_segment**2))
            peak = np.max(np.abs(audio_segment))
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
            
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sample_rate))
            
            fft = np.fft.rfft(audio_segment)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_segment), 1/self.sample_rate)
            dominant_freq = freqs[np.argmax(magnitude)]
            
            self.current_audio_metrics = {
                'rms': rms,
                'peak': peak,
                'zcr': zcr,
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'dominant_freq': dominant_freq
            }
            
        except Exception as e:
            print(f"Metrics error: {e}")
    
    def update_audio_metrics_display(self):
        """Update audio metrics display"""
        m = self.current_audio_metrics
        
        self.rms_value.config(text=f"{m['rms']:.4f}")
        self.peak_value.config(text=f"{m['peak']:.4f}")
        self.zcr_value.config(text=f"{m['zcr']:.4f}")
        self.dominant_freq_value.config(text=f"{m['dominant_freq']:.0f} Hz")
        self.centroid_value.config(text=f"{m['spectral_centroid']:.0f} Hz")
        self.rolloff_value.config(text=f"{m['spectral_rolloff']:.0f} Hz")
        self.bandwidth_value.config(text=f"{m['spectral_bandwidth']:.0f} Hz")
    
    def calculate_threshold_confidence(self, normal_prob):
        """Calculate confidence score"""
        actual_confidence = normal_prob * 100
        
        if actual_confidence >= self.THRESHOLD:
            confidence = 100 - (100 - actual_confidence)
        else:
            confidence = 100 - (self.THRESHOLD - actual_confidence)
        
        return confidence
    
    def extract_features(self, audio_segment):
        """Extract acoustic features"""
        try:
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sample_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sample_rate))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=self.sample_rate)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_segment))
            rms = np.mean(librosa.feature.rms(y=audio_segment))
            
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
            print(f"Feature extraction error: {e}")
            return None
    
    def run(self):
        """Start the application"""
        self.window.mainloop()


if __name__ == "__main__":
    app = StenosisTester()
    app.run()