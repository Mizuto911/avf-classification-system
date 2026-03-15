# coding: utf-8
import tkinter as tk
from tkinter import messagebox
import librosa
import numpy as np
from pathlib import Path
import threading
import warnings
import sounddevice as sd
import queue
import time

# Matplotlib for visualization
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

warnings.filterwarnings('ignore')

class StenosisTester:
    # ============================================
    # DETECTION THRESHOLDS - Adjust based on clinical validation
    # ============================================
    
    # Primary indicators
    STENOSIS_FREQ_THRESHOLD = 500  # Hz - stenosis typically > 500 Hz
    NORMAL_FREQ_MAX = 350  # Hz - normal typically < 350 Hz
    
    # Secondary indicators
    SPECTRAL_CENTROID_THRESHOLD = 600  # Hz
    SPECTRAL_BANDWIDTH_THRESHOLD = 800  # Hz - wide bandwidth = turbulent
    ZCR_THRESHOLD = 0.15  # Zero crossing rate - high = high frequency
    
    # Confidence threshold for classification
    MIN_CONFIDENCE_THRESHOLD = 60.0  # If confidence < 60%, flip the prediction
    
    # Scoring weights
    WEIGHTS = {
        'dominant_freq': 0.35,
        'spectral_centroid': 0.25,
        'spectral_bandwidth': 0.20,
        'zcr': 0.20
    }
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AVF Stenosis Detector")
        
        # FIXED SIZE FOR 800x480 SCREEN
        self.window.overrideredirect(True)
        self.window.geometry("800x480+0+0")
        self.window.configure(bg='#1a1a1a')
        
        # Audio parameters
        self.sample_rate = 22050
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
    
    def analyze_stenosis(self, metrics):
        """
        Rule-based stenosis detection using audio metrics
        Returns: (prediction, confidence, score_breakdown)
        """
        scores = {}
        
        # 1. Dominant Frequency Analysis
        if metrics['dominant_freq'] > self.STENOSIS_FREQ_THRESHOLD:
            freq_score = min(1.0, (metrics['dominant_freq'] - self.STENOSIS_FREQ_THRESHOLD) / 300)
        else:
            freq_score = 0.0
        scores['dominant_freq'] = freq_score
        
        # 2. Spectral Centroid Analysis
        if metrics['spectral_centroid'] > self.SPECTRAL_CENTROID_THRESHOLD:
            centroid_score = min(1.0, (metrics['spectral_centroid'] - self.SPECTRAL_CENTROID_THRESHOLD) / 400)
        else:
            centroid_score = 0.0
        scores['spectral_centroid'] = centroid_score
        
        # 3. Spectral Bandwidth Analysis
        if metrics['spectral_bandwidth'] > self.SPECTRAL_BANDWIDTH_THRESHOLD:
            bandwidth_score = min(1.0, (metrics['spectral_bandwidth'] - self.SPECTRAL_BANDWIDTH_THRESHOLD) / 500)
        else:
            bandwidth_score = 0.0
        scores['spectral_bandwidth'] = bandwidth_score
        
        # 4. Zero Crossing Rate
        if metrics['zcr'] > self.ZCR_THRESHOLD:
            zcr_score = min(1.0, (metrics['zcr'] - self.ZCR_THRESHOLD) / 0.1)
        else:
            zcr_score = 0.0
        scores['zcr'] = zcr_score
        
        # Calculate weighted stenosis score
        stenosis_score = (
            scores['dominant_freq'] * self.WEIGHTS['dominant_freq'] +
            scores['spectral_centroid'] * self.WEIGHTS['spectral_centroid'] +
            scores['spectral_bandwidth'] * self.WEIGHTS['spectral_bandwidth'] +
            scores['zcr'] * self.WEIGHTS['zcr']
        )
        
        # Determine prediction with confidence threshold logic
        if stenosis_score > 0.5:
            prediction = "STENOSIS"
            confidence = stenosis_score * 100
        else:
            prediction = "NORMAL"
            confidence = (1 - stenosis_score) * 100
        
        # CRITICAL FIX: If confidence is too low, flip the prediction
        if confidence < self.MIN_CONFIDENCE_THRESHOLD:
            if prediction == "NORMAL":
                prediction = "STENOSIS"
                confidence = stenosis_score * 100
            else:
                prediction = "NORMAL"
                confidence = (1 - stenosis_score) * 100
        
        return prediction, confidence, scores
    
    def setup_ui(self):
        # Very compact header bar - 25px
        header_frame = tk.Frame(self.window, bg='#0d47a1', height=25)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="AVF Stenosis Detector",
            font=("Arial", 9, "bold"),
            bg='#0d47a1',
            fg='white'
        )
        title_label.pack(side='left', padx=5, pady=2)
        
        # Close button
        close_btn = tk.Button(
            header_frame,
            text="X",
            command=self.window.quit,
            font=("Arial", 10, "bold"),
            bg='#d32f2f',
            fg='white',
            activebackground='#b71c1c',
            cursor="hand2",
            relief='flat',
            width=2
        )
        close_btn.pack(side='right', padx=5)
        
        # Main container - 455px available (480 - 25 header)
        main_container = tk.Frame(self.window, bg='#1a1a1a')
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Control row - 35px
        control_frame = tk.Frame(main_container, bg='#2c2c2c', relief='solid', borderwidth=1, height=35)
        control_frame.pack(fill='x', pady=(0, 3))
        control_frame.pack_propagate(False)
        
        control_inner = tk.Frame(control_frame, bg='#2c2c2c')
        control_inner.pack(fill='both', expand=True, padx=5, pady=3)
        
        # Start button
        self.start_btn = tk.Button(
            control_inner,
            text="START",
            command=self.toggle_listening,
            font=("Arial", 9, "bold"),
            bg='#4caf50',
            fg='white',
            activebackground='#45a049',
            cursor="hand2",
            relief='flat',
            width=8
        )
        self.start_btn.pack(side='left', padx=(0, 8))
        
        # Status
        self.status_indicator = tk.Label(
            control_inner,
            text="Ready",
            font=("Arial", 8, "bold"),
            bg='#2c2c2c',
            fg='#9e9e9e'
        )
        self.status_indicator.pack(side='left', padx=(0, 15))
        
        # Result
        tk.Label(
            control_inner,
            text="Result:",
            font=("Arial", 7, "bold"),
            bg='#2c2c2c',
            fg='#bdbdbd'
        ).pack(side='left', padx=(0, 5))
        
        self.result_label = tk.Label(
            control_inner,
            text="READY",
            font=("Arial", 12, "bold"),
            bg='#2c2c2c',
            fg='#9e9e9e'
        )
        self.result_label.pack(side='left', padx=(0, 10))
        
        # Confidence
        tk.Label(
            control_inner,
            text="Conf:",
            font=("Arial", 7, "bold"),
            bg='#2c2c2c',
            fg='#bdbdbd'
        ).pack(side='left', padx=(0, 5))
        
        self.confidence_label = tk.Label(
            control_inner,
            text="--",
            font=("Arial", 10, "bold"),
            bg='#2c2c2c',
            fg='#ffffff'
        )
        self.confidence_label.pack(side='left')
        
        # Visualization - 200px height
        viz_frame = tk.Frame(main_container, bg='#2c2c2c', relief='solid', borderwidth=1, height=200)
        viz_frame.pack(fill='x', pady=(0, 3))
        viz_frame.pack_propagate(False)
        
        # Create matplotlib figure - VERY COMPACT for 800x480
        self.fig = Figure(figsize=(9.5, 2.2), facecolor='#2c2c2c', dpi=80)
        
        # Waveform plot (LEFT)
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax1.set_facecolor('#1a1a1a')
        self.ax1.set_title('Waveform', color='white', fontsize=8, pad=3)
        self.ax1.set_xlabel('Samples', color='white', fontsize=6)
        self.ax1.set_ylabel('Amp', color='white', fontsize=6)
        self.ax1.tick_params(colors='white', labelsize=5)
        self.line1, = self.ax1.plot([], [], color='#00bcd4', linewidth=1)
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True, alpha=0.2, color='white')
        
        # Spectrum plot (RIGHT)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        self.ax2.set_facecolor('#1a1a1a')
        self.ax2.set_title('Frequency (0-2kHz)', color='white', fontsize=8, pad=3)
        self.ax2.set_xlabel('Hz', color='white', fontsize=6)
        self.ax2.set_ylabel('Mag', color='white', fontsize=6)
        self.ax2.tick_params(colors='white', labelsize=5)
        self.line2, = self.ax2.plot([], [], color='#ff5722', linewidth=1)
        self.ax2.set_xlim(0, 2000)
        self.ax2.grid(True, alpha=0.2, color='white')
        
        # Add threshold markers
        self.ax2.axvline(x=self.STENOSIS_FREQ_THRESHOLD, color='#f44336', linestyle='--', 
                        linewidth=1, alpha=0.5, label=f'Sten')
        self.ax2.axvline(x=self.NORMAL_FREQ_MAX, color='#4caf50', linestyle='--', 
                        linewidth=1, alpha=0.5, label=f'Norm')
        self.ax2.legend(fontsize=5, loc='upper right', facecolor='#2c2c2c', edgecolor='white', labelcolor='white')
        
        self.fig.tight_layout(pad=0.5)
        
        # Embed canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Data area - 2 columns - 212px remaining
        data_frame = tk.Frame(main_container, bg='#1a1a1a', height=212)
        data_frame.pack(fill='both', expand=True)
        data_frame.pack_propagate(False)
        
        # LEFT COLUMN - Audio Metrics
        left_column = tk.Frame(data_frame, bg='#2c2c2c', relief='solid', borderwidth=1)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 2))
        
        # Header
        tk.Label(
            left_column,
            text="AUDIO METRICS",
            font=("Arial", 8, "bold"),
            bg='#2c2c2c',
            fg='#ffffff',
            pady=2
        ).pack(fill='x')
        
        metrics_container = tk.Frame(left_column, bg='#2c2c2c')
        metrics_container.pack(fill='both', expand=True, padx=5, pady=2)
        
        self.create_metric_row(metrics_container, "RMS", "rms")
        self.create_metric_row(metrics_container, "Peak", "peak")
        self.create_metric_row(metrics_container, "ZCR", "zcr")
        self.create_metric_row(metrics_container, "Dom Freq", "dominant_freq")
        self.create_metric_row(metrics_container, "Centroid", "centroid")
        self.create_metric_row(metrics_container, "Rolloff", "rolloff")
        self.create_metric_row(metrics_container, "Bandwidth", "bandwidth")
        
        # RIGHT COLUMN - Detection Scores
        right_column = tk.Frame(data_frame, bg='#2c2c2c', relief='solid', borderwidth=1)
        right_column.pack(side='right', fill='both', expand=True, padx=(2, 0))
        
        # Header
        tk.Label(
            right_column,
            text="STENOSIS INDICATORS",
            font=("Arial", 8, "bold"),
            bg='#2c2c2c',
            fg='#ffffff',
            pady=2
        ).pack(fill='x')
        
        scores_container = tk.Frame(right_column, bg='#2c2c2c')
        scores_container.pack(fill='both', expand=True, padx=5, pady=2)
        
        self.create_metric_row(scores_container, "Freq", "freq_score")
        self.create_metric_row(scores_container, "Centroid", "centroid_score")
        self.create_metric_row(scores_container, "Bandwidth", "bandwidth_score")
        self.create_metric_row(scores_container, "ZCR", "zcr_score")
        tk.Frame(scores_container, bg='#2c2c2c', height=5).pack()
        self.create_metric_row(scores_container, "Overall", "overall_score")
        self.create_metric_row(scores_container, "Samples", "samples")
    
    def create_metric_row(self, parent, label_text, key):
        """Create a compact metric row"""
        frame = tk.Frame(parent, bg='#2c2c2c')
        frame.pack(fill='x', pady=1)
        
        # Label
        label = tk.Label(
            frame,
            text=f"{label_text}:",
            font=("Arial", 7, "bold"),
            bg='#2c2c2c',
            fg='#bdbdbd',
            anchor='w'
        )
        label.pack(side='left', fill='x', expand=True)
        
        # Value
        value = tk.Label(
            frame,
            text="--",
            font=("Arial", 7),
            bg='#2c2c2c',
            fg='#4caf50',
            anchor='e',
            width=12
        )
        value.pack(side='right')
        
        # Store reference
        setattr(self, f"{key}_value", value)
    
    def toggle_listening(self):
        """Start or stop listening"""
        if not self.is_live_detecting:
            self.start_listening()
        else:
            self.stop_listening()
    
    def start_listening(self):
        """Start live detection"""
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
            text="STOP",
            bg='#f44336',
            activebackground='#d32f2f'
        )
        
        self.status_indicator.config(
            text="Listening",
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
                    
                    # Analyze for stenosis using rule-based method
                    prediction, confidence, scores = self.analyze_stenosis(self.current_audio_metrics)
                    
                    self.live_predictions.append((prediction, confidence, scores))
                    
                    if len(self.live_predictions) > self.live_prediction_window:
                        self.live_predictions = self.live_predictions[-self.live_prediction_window:]
                    
                    self.window.after(0, self._update_results, prediction, confidence, scores, len(self.live_predictions))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def _update_results(self, prediction, confidence, scores, num_samples):
        """Update results display"""
        if not self.is_live_detecting:
            return
        
        # Color coding
        if prediction == 'STENOSIS':
            color = "#f44336"
            status_color = "#f44336"
            status_text = "Stenosis"
        else:
            color = "#4caf50"
            status_color = "#4caf50"
            status_text = "Normal"
        
        self.result_label.config(text=prediction, fg=color)
        self.confidence_label.config(text=f"{confidence:.1f}%")
        self.status_indicator.config(text=status_text, fg=status_color)
        
        # Update scores
        self.freq_score_value.config(text=f"{scores['dominant_freq']*100:.1f}%")
        self.centroid_score_value.config(text=f"{scores['spectral_centroid']*100:.1f}%")
        self.bandwidth_score_value.config(text=f"{scores['spectral_bandwidth']*100:.1f}%")
        self.zcr_score_value.config(text=f"{scores['zcr']*100:.1f}%")
        
        # Overall stenosis score
        overall = (scores['dominant_freq'] + scores['spectral_centroid'] + 
                  scores['spectral_bandwidth'] + scores['zcr']) / 4
        self.overall_score_value.config(text=f"{overall*100:.1f}%")
        
        self.samples_value.config(text=f"{num_samples}")
    
    def stop_listening(self):
        """Stop live detection"""
        self.is_live_detecting = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.start_btn.config(
            text="START",
            bg='#4caf50',
            activebackground='#45a049'
        )
        
        self.status_indicator.config(
            text="Ready",
            fg='#9e9e9e'
        )
    
    def update_visualization(self, audio_data):
        """Update visualization plots"""
        try:
            # Waveform
            if len(audio_data) > 500:
                waveform_data = audio_data[-500:]
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
        self.dominant_freq_value.config(text=f"{m['dominant_freq']:.0f}Hz")
        self.centroid_value.config(text=f"{m['spectral_centroid']:.0f}Hz")
        self.rolloff_value.config(text=f"{m['spectral_rolloff']:.0f}Hz")
        self.bandwidth_value.config(text=f"{m['spectral_bandwidth']:.0f}Hz")
    
    def run(self):
        """Start the application"""
        self.window.mainloop()


if __name__ == "__main__":
    app = StenosisTester()
    app.run()