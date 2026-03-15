# coding: utf-8
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import librosa
import numpy as np
import joblib
from pathlib import Path
import threading
import warnings
import sounddevice as sd
import soundfile as sf
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
    # ============================================
    # CONFIGURATION - Easy to adjust
    # ============================================
    THRESHOLD = 90.0  # Classification threshold (% normal probability)
    # To adjust sensitivity:
    # - Higher (e.g., 95) = Less sensitive, fewer stenosis alerts
    # - Lower (e.g., 85) = More sensitive, more stenosis alerts
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AVF Stenosis Detector")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg='#2c3e50')
        
        # Model parameters
        self.model = None
        self.scaler = None
        self.segment_length = 3
        self.hop_length = 2
        self.sample_rate = 22050
        self.selected_file = None
        
        # Recording parameters - AUTO-DETECT USB SOUND CARD
        self.is_recording = False
        self.recording_data = None
        self.mic_device_id = None
        self.recording_duration = 30
        self.mic_sample_rate = 48000
        
        # Auto-detect USB audio device
        self.detect_usb_audio()
        
        # Live detection parameters
        self.is_live_detecting = False
        self.audio_queue = queue.Queue()
        self.live_buffer = []
        self.live_buffer_size = self.mic_sample_rate * 3  # 3 seconds
        self.live_predictions = []
        self.live_prediction_window = 10  # Keep last 10 predictions
        
        # Visualization data
        self.visualization_data = {
            'waveform': np.zeros(1000),
            'spectrum': np.zeros(512),
            'rms_history': [],
            'dominant_freq': 0,
            'spectral_centroid': 0
        }
        
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
        """Auto-detect USB audio device (AB13X USB Audio)"""
        try:
            devices = sd.query_devices()
            
            # Look for USB audio device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    # Check for USB audio keywords
                    if 'usb' in device_name or 'ab13x' in device_name:
                        self.mic_device_id = i
                        self.mic_sample_rate = int(device['default_samplerate'])
                        self.device_name = device['name']
                        self.device_channels = device['max_input_channels']
                        print(f"Found USB audio device: {device['name']} (Device {i})")
                        return
            
            # If no USB device found, use first available input device
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.mic_device_id = i
                    self.mic_sample_rate = int(device['default_samplerate'])
                    self.device_name = device['name']
                    self.device_channels = device['max_input_channels']
                    print(f"Using first available device: {device['name']} (Device {i})")
                    return
            
            # No input devices found
            self.device_name = "No input device found"
            self.device_channels = 0
            print("WARNING: No audio input devices detected!")
            
        except Exception as e:
            print(f"Error detecting audio device: {e}")
            self.device_name = "Error detecting device"
            self.device_channels = 0
    
    def create_visualization_panel(self, parent_frame):
        """Create real-time audio visualization"""
        viz_frame = tk.Frame(parent_frame, bg='#34495e', relief='solid', borderwidth=2)
        viz_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        # Title
        viz_title = tk.Label(
            viz_frame,
            text="AUDIO VISUALIZATION",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='#ecf0f1'
        )
        viz_title.pack(pady=(8, 5))
        
        # Create matplotlib figure with 2 subplots
        self.fig = Figure(figsize=(5, 4), facecolor='#34495e')
        
        # Waveform plot
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax1.set_facecolor('#2c3e50')
        self.ax1.set_title('Waveform', color='white', fontsize=9)
        self.ax1.set_ylabel('Amplitude', color='white', fontsize=8)
        self.ax1.tick_params(colors='white', labelsize=7)
        self.line1, = self.ax1.plot([], [], color='#3498db', linewidth=1)
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True, alpha=0.3, color='white')
        
        # Spectrum plot
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.ax2.set_facecolor('#2c3e50')
        self.ax2.set_title('Frequency Spectrum', color='white', fontsize=9)
        self.ax2.set_xlabel('Frequency (Hz)', color='white', fontsize=8)
        self.ax2.set_ylabel('Magnitude', color='white', fontsize=8)
        self.ax2.tick_params(colors='white', labelsize=7)
        self.line2, = self.ax2.plot([], [], color='#e74c3c', linewidth=1)
        self.ax2.set_xlim(0, 2000)  # Focus on 0-2kHz
        self.ax2.grid(True, alpha=0.3, color='white')
        
        self.fig.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Audio metrics display
        metrics_frame = tk.Frame(viz_frame, bg='#34495e')
        metrics_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # RMS Level
        rms_label = tk.Label(
            metrics_frame,
            text="RMS Level:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7'
        )
        rms_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        self.rms_value = tk.Label(
            metrics_frame,
            text="0.000",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.rms_value.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Dominant Frequency
        freq_label = tk.Label(
            metrics_frame,
            text="Dominant Freq:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7'
        )
        freq_label.grid(row=0, column=2, sticky='w', padx=5, pady=2)
        
        self.freq_value = tk.Label(
            metrics_frame,
            text="0 Hz",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.freq_value.grid(row=0, column=3, sticky='w', padx=5, pady=2)
        
        # Spectral Centroid
        centroid_label = tk.Label(
            metrics_frame,
            text="Centroid:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7'
        )
        centroid_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        self.centroid_value = tk.Label(
            metrics_frame,
            text="0 Hz",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.centroid_value.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Peak Level
        peak_label = tk.Label(
            metrics_frame,
            text="Peak:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7'
        )
        peak_label.grid(row=1, column=2, sticky='w', padx=5, pady=2)
        
        self.peak_value = tk.Label(
            metrics_frame,
            text="0.000",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.peak_value.grid(row=1, column=3, sticky='w', padx=5, pady=2)
    
    def update_visualization(self, audio_data):
        """Update visualization with new audio data"""
        try:
            # Update waveform (show last 1000 samples)
            if len(audio_data) > 1000:
                waveform_data = audio_data[-1000:]
            else:
                waveform_data = audio_data
            
            # Update waveform plot
            x_wave = np.arange(len(waveform_data))
            self.line1.set_data(x_wave, waveform_data)
            self.ax1.set_xlim(0, len(waveform_data))
            
            # Calculate and update spectrum
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.mic_sample_rate)
            
            # Focus on 0-2kHz range
            freq_mask = freqs <= 2000
            freqs_display = freqs[freq_mask]
            magnitude_display = magnitude[freq_mask]
            
            # Update spectrum plot
            self.line2.set_data(freqs_display, magnitude_display)
            max_mag = np.max(magnitude_display)
            self.ax2.set_ylim(0, max_mag * 1.1 if max_mag > 0 else 1)
            
            # Calculate metrics
            rms = np.sqrt(np.mean(audio_data**2))
            peak = np.max(np.abs(audio_data))
            dominant_freq_idx = np.argmax(magnitude)
            dominant_freq = freqs[dominant_freq_idx]
            
            # Calculate spectral centroid
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0
            
            # Update metric displays
            self.rms_value.config(text=f"{rms:.4f}")
            self.peak_value.config(text=f"{peak:.4f}")
            self.freq_value.config(text=f"{dominant_freq:.0f} Hz")
            self.centroid_value.config(text=f"{spectral_centroid:.0f} Hz")
            
            # Redraw canvas
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def extract_and_store_metrics(self, audio_segment):
        """Extract and store audio metrics for display"""
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
            
            # Store metrics
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
            print(f"Error extracting metrics: {e}")
        
    def setup_ui(self):
        # Header with close button
        header_frame = tk.Frame(self.window, bg='#34495e', height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Close button (top right)
        close_btn = tk.Button(
            header_frame,
            text="X",
            command=self.window.quit,
            font=("Arial", 16, "bold"),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            cursor="hand2",
            relief='flat',
            width=3,
            height=1
        )
        close_btn.pack(side='right', padx=10, pady=10)
        
        title_label = tk.Label(
            header_frame, 
            text="AVF Stenosis Detector", 
            font=("Arial", 20, "bold"),
            bg='#34495e',
            fg='white'
        )
        title_label.pack(side='left', padx=20, pady=15)
        
        # Main content area - split into left and right
        content_frame = tk.Frame(self.window, bg='#2c3e50')
        content_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # LEFT SIDE - Controls (40% width)
        left_frame = tk.Frame(content_frame, bg='#2c3e50')
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # LIVE DETECTION section title
        live_title = tk.Label(
            left_frame,
            text="LIVE DETECTION",
            font=("Arial", 10, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        live_title.pack(anchor='w', pady=(0, 3))
        
        # Live detect button
        self.live_detect_btn = tk.Button(
            left_frame,
            text="START LIVE DETECTION",
            command=self.toggle_live_detection,
            font=("Arial", 11, "bold"),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            activeforeground='white',
            cursor="hand2",
            relief='flat',
            borderwidth=0,
            pady=8
        )
        self.live_detect_btn.pack(fill='x', pady=(0, 8))
        
        # Separator
        separator0 = tk.Frame(left_frame, bg='#7f8c8d', height=1)
        separator0.pack(fill='x', pady=(0, 8))
        
        # Recording section title
        recording_title = tk.Label(
            left_frame,
            text="RECORD & ANALYZE",
            font=("Arial", 10, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        recording_title.pack(anchor='w', pady=(0, 3))
        
        # Record button
        self.record_btn = tk.Button(
            left_frame,
            text="RECORD (30s)",
            command=self.toggle_recording,
            font=("Arial", 11, "bold"),
            bg='#9b59b6',
            fg='white',
            activebackground='#8e44ad',
            activeforeground='white',
            cursor="hand2",
            relief='flat',
            borderwidth=0,
            pady=8
        )
        self.record_btn.pack(fill='x', pady=(0, 8))
        
        # Separator
        separator1 = tk.Frame(left_frame, bg='#7f8c8d', height=1)
        separator1.pack(fill='x', pady=(0, 8))
        
        # File selection section title
        file_title = tk.Label(
            left_frame,
            text="OR SELECT FILE",
            font=("Arial", 10, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        file_title.pack(anchor='w', pady=(0, 3))
        
        # File display
        file_display_frame = tk.Frame(left_frame, bg='white', relief='solid', borderwidth=2)
        file_display_frame.pack(fill='x', pady=(0, 5))
        
        self.file_label = tk.Label(
            file_display_frame,
            text="No file selected",
            font=("Arial", 9),
            bg='white',
            fg='#7f8c8d',
            anchor='w',
            padx=8,
            pady=6,
            wraplength=280
        )
        self.file_label.pack(fill='both')
        
        # Browse button
        browse_btn = tk.Button(
            left_frame,
            text="BROWSE FILE",
            command=self.select_file,
            font=("Arial", 11, "bold"),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            cursor="hand2",
            relief='flat',
            borderwidth=0,
            pady=8
        )
        browse_btn.pack(fill='x', pady=(0, 8))
        
        # Separator
        separator2 = tk.Frame(left_frame, bg='#7f8c8d', height=1)
        separator2.pack(fill='x', pady=(0, 8))
        
        # Detect button
        self.detect_btn = tk.Button(
            left_frame,
            text="ANALYZE FILE",
            command=self.analyze_recording,
            font=("Arial", 11, "bold"),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            activeforeground='white',
            cursor="hand2",
            state="disabled",
            relief='flat',
            borderwidth=0,
            pady=8
        )
        self.detect_btn.pack(fill='x', pady=(0, 8))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            left_frame,
            length=300,
            mode='indeterminate'
        )
        self.progress.pack(fill='x', pady=(0, 10))
        
        # Model status info
        status_info_frame = tk.Frame(left_frame, bg='#34495e', relief='solid', borderwidth=1)
        status_info_frame.pack(fill='x', pady=(0, 10))
        
        info_title = tk.Label(
            status_info_frame,
            text="SYSTEM STATUS",
            font=("Arial", 10, "bold"),
            bg='#34495e',
            fg='#ecf0f1'
        )
        info_title.pack(pady=(8, 5))
        
        self.info_label = tk.Label(
            status_info_frame,
            text="Initializing...",
            font=("Arial", 9),
            bg='#34495e',
            fg='#bdc3c7',
            wraplength=280,
            justify='left'
        )
        self.info_label.pack(padx=10, pady=(0, 8))
        
        # Add visualization panel
        self.create_visualization_panel(left_frame)
        
        # RIGHT SIDE - Results (60% width)
        right_frame = tk.Frame(content_frame, bg='#34495e', relief='solid', borderwidth=2)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Result header
        result_header = tk.Label(
            right_frame,
            text="ANALYSIS RESULTS",
            font=("Arial", 13, "bold"),
            bg='#34495e',
            fg='#ecf0f1'
        )
        result_header.pack(pady=(12, 8))
        
        # Main result
        self.result_label = tk.Label(
            right_frame,
            text="Ready",
            font=("Arial", 40, "bold"),
            bg='#34495e',
            fg='#95a5a6'
        )
        self.result_label.pack(pady=(20, 20))
        
        # Separator line
        separator = tk.Frame(right_frame, bg='#7f8c8d', height=2)
        separator.pack(fill='x', padx=20, pady=10)
        
        # Details section
        details_container = tk.Frame(right_frame, bg='#34495e')
        details_container.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        
        # Duration row
        duration_frame = tk.Frame(details_container, bg='#34495e')
        duration_frame.pack(fill='x', pady=2)
        
        duration_label_text = tk.Label(
            duration_frame,
            text="Duration:",
            font=("Arial", 10, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=14,
            anchor='w'
        )
        duration_label_text.pack(side='left')
        
        self.duration_value = tk.Label(
            duration_frame,
            text="--",
            font=("Arial", 10),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.duration_value.pack(side='left', fill='x', expand=True)
        
        # Segments row
        segments_frame = tk.Frame(details_container, bg='#34495e')
        segments_frame.pack(fill='x', pady=2)
        
        segments_label_text = tk.Label(
            segments_frame,
            text="Segments:",
            font=("Arial", 10, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=14,
            anchor='w'
        )
        segments_label_text.pack(side='left')
        
        self.segments_value = tk.Label(
            segments_frame,
            text="--",
            font=("Arial", 10),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.segments_value.pack(side='left', fill='x', expand=True)
        
        # Consistency row
        consistency_frame = tk.Frame(details_container, bg='#34495e')
        consistency_frame.pack(fill='x', pady=2)
        
        consistency_label_text = tk.Label(
            consistency_frame,
            text="Consistency:",
            font=("Arial", 10, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=14,
            anchor='w'
        )
        consistency_label_text.pack(side='left')
        
        self.consistency_value = tk.Label(
            consistency_frame,
            text="--",
            font=("Arial", 10),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.consistency_value.pack(side='left', fill='x', expand=True)
        
        # Confidence Score row
        confidence_score_frame = tk.Frame(details_container, bg='#34495e')
        confidence_score_frame.pack(fill='x', pady=2)
        
        confidence_score_label_text = tk.Label(
            confidence_score_frame,
            text="Confidence:",
            font=("Arial", 10, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=14,
            anchor='w'
        )
        confidence_score_label_text.pack(side='left')
        
        self.confidence_score_value = tk.Label(
            confidence_score_frame,
            text="--",
            font=("Arial", 10),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.confidence_score_value.pack(side='left', fill='x', expand=True)
        
        # Audio Metrics Section Header
        audio_metrics_separator = tk.Frame(right_frame, bg='#7f8c8d', height=2)
        audio_metrics_separator.pack(fill='x', padx=20, pady=10)
        
        audio_metrics_header = tk.Label(
            right_frame,
            text="AUDIO METRICS",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='#ecf0f1'
        )
        audio_metrics_header.pack(pady=(0, 8))
        
        # Audio metrics container
        audio_metrics_container = tk.Frame(right_frame, bg='#34495e')
        audio_metrics_container.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        
        # RMS Energy
        rms_frame = tk.Frame(audio_metrics_container, bg='#34495e')
        rms_frame.pack(fill='x', pady=2)
        
        rms_label = tk.Label(
            rms_frame,
            text="RMS Energy:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=16,
            anchor='w'
        )
        rms_label.pack(side='left')
        
        self.audio_rms_value = tk.Label(
            rms_frame,
            text="--",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.audio_rms_value.pack(side='left', fill='x', expand=True)
        
        # Peak Amplitude
        peak_frame = tk.Frame(audio_metrics_container, bg='#34495e')
        peak_frame.pack(fill='x', pady=2)
        
        peak_label = tk.Label(
            peak_frame,
            text="Peak Amplitude:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=16,
            anchor='w'
        )
        peak_label.pack(side='left')
        
        self.audio_peak_value = tk.Label(
            peak_frame,
            text="--",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.audio_peak_value.pack(side='left', fill='x', expand=True)
        
        # Zero Crossing Rate
        zcr_frame = tk.Frame(audio_metrics_container, bg='#34495e')
        zcr_frame.pack(fill='x', pady=2)
        
        zcr_label = tk.Label(
            zcr_frame,
            text="Zero Cross Rate:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=16,
            anchor='w'
        )
        zcr_label.pack(side='left')
        
        self.audio_zcr_value = tk.Label(
            zcr_frame,
            text="--",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.audio_zcr_value.pack(side='left', fill='x', expand=True)
        
        # Dominant Frequency
        dom_freq_frame = tk.Frame(audio_metrics_container, bg='#34495e')
        dom_freq_frame.pack(fill='x', pady=2)
        
        dom_freq_label = tk.Label(
            dom_freq_frame,
            text="Dominant Freq:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=16,
            anchor='w'
        )
        dom_freq_label.pack(side='left')
        
        self.audio_dom_freq_value = tk.Label(
            dom_freq_frame,
            text="--",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.audio_dom_freq_value.pack(side='left', fill='x', expand=True)
        
        # Spectral Centroid
        centroid_frame = tk.Frame(audio_metrics_container, bg='#34495e')
        centroid_frame.pack(fill='x', pady=2)
        
        centroid_label = tk.Label(
            centroid_frame,
            text="Spectral Centroid:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=16,
            anchor='w'
        )
        centroid_label.pack(side='left')
        
        self.audio_centroid_value = tk.Label(
            centroid_frame,
            text="--",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.audio_centroid_value.pack(side='left', fill='x', expand=True)
        
        # Spectral Rolloff
        rolloff_frame = tk.Frame(audio_metrics_container, bg='#34495e')
        rolloff_frame.pack(fill='x', pady=2)
        
        rolloff_label = tk.Label(
            rolloff_frame,
            text="Spectral Rolloff:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=16,
            anchor='w'
        )
        rolloff_label.pack(side='left')
        
        self.audio_rolloff_value = tk.Label(
            rolloff_frame,
            text="--",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.audio_rolloff_value.pack(side='left', fill='x', expand=True)
        
        # Spectral Bandwidth
        bandwidth_frame = tk.Frame(audio_metrics_container, bg='#34495e')
        bandwidth_frame.pack(fill='x', pady=2)
        
        bandwidth_label = tk.Label(
            bandwidth_frame,
            text="Spectral Bandwidth:",
            font=("Arial", 9, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=16,
            anchor='w'
        )
        bandwidth_label.pack(side='left')
        
        self.audio_bandwidth_value = tk.Label(
            bandwidth_frame,
            text="--",
            font=("Arial", 9),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.audio_bandwidth_value.pack(side='left', fill='x', expand=True)
        
        # Status bar
        self.status_label = tk.Label(
            self.window,
            text="Initializing...",
            font=("Arial", 10, "bold"),
            bg='#1abc9c',
            fg='white',
            anchor='w',
            padx=15,
            pady=8
        )
        self.status_label.pack(side='bottom', fill='x')
    
    def update_audio_metrics_display(self):
        """Update the audio metrics display on right panel"""
        metrics = self.current_audio_metrics
        
        self.audio_rms_value.config(text=f"{metrics['rms']:.4f}")
        self.audio_peak_value.config(text=f"{metrics['peak']:.4f}")
        self.audio_zcr_value.config(text=f"{metrics['zcr']:.4f}")
        self.audio_dom_freq_value.config(text=f"{metrics['dominant_freq']:.1f} Hz")
        self.audio_centroid_value.config(text=f"{metrics['spectral_centroid']:.1f} Hz")
        self.audio_rolloff_value.config(text=f"{metrics['spectral_rolloff']:.1f} Hz")
        self.audio_bandwidth_value.config(text=f"{metrics['spectral_bandwidth']:.1f} Hz")
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent.absolute()
            model_path = script_dir / 'stenosis_model.pkl'
            scaler_path = script_dir / 'scaler.pkl'
            
            self.model = joblib.load(str(model_path))
            self.scaler = joblib.load(str(scaler_path))
            
            # Show device info
            device_info = f"Audio Device:\n{self.device_name}\n{self.mic_sample_rate} Hz"
            if self.mic_device_id is None:
                device_status = "WARNING: No input device"
                status_color = '#e74c3c'
            else:
                device_status = f"Device {self.mic_device_id}: {self.device_name}"
                status_color = '#27ae60'
            
            self.status_label.config(
                text=f"Model loaded - {device_status}",
                bg=status_color
            )
            self.info_label.config(
                text=f"Model: Loaded\nScaler: Loaded\n\n{device_info}\n\nReady to start detection\nor record audio."
            )
        except FileNotFoundError:
            self.status_label.config(
                text="ERROR: Model files not found",
                bg='#e74c3c'
            )
            self.info_label.config(
                text="ERROR: Cannot find model files\n\nLooking in:\n" + 
                     str(Path(__file__).parent.absolute())
            )
            messagebox.showerror(
                "Model Not Found",
                "Cannot find model files in:\n" + str(Path(__file__).parent.absolute()) +
                "\n\nMake sure stenosis_model.pkl and scaler.pkl are present."
            )
        except Exception as e:
            self.status_label.config(
                text=f"ERROR: {str(e)[:40]}",
                bg='#e74c3c'
            )
            self.info_label.config(
                text=f"ERROR: {str(e)}"
            )
    
    def toggle_live_detection(self):
        """Start or stop live detection"""
        if not self.is_live_detecting:
            self.start_live_detection()
        else:
            self.stop_live_detection()
    
    def start_live_detection(self):
        """Start live real-time detection"""
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return
        
        if self.mic_device_id is None:
            messagebox.showerror("No Device", "No audio input device detected")
            return
        
        self.is_live_detecting = True
        
        # CRITICAL: Clear all buffers and queue to start fresh
        self.live_buffer = []
        self.live_predictions = []
        
        # Clear the audio queue completely
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Update UI
        self.live_detect_btn.config(
            text="STOP LIVE DETECTION",
            bg='#95a5a6',
            activebackground='#7f8c8d'
        )
        self.record_btn.config(state="disabled")
        self.detect_btn.config(state="disabled", bg='#95a5a6')
        
        self.clear_results()
        self.result_label.config(text="Listening...", fg="#f39c12")
        
        self.status_label.config(
            text="Live detection active - Analyzing audio in real-time",
            bg='#f39c12'
        )
        self.info_label.config(
            text="Live Detection Active\n\nListening continuously...\nProcessing every 3 seconds\n\nPlace stethoscope on\nfistula site"
        )
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                samplerate=self.mic_sample_rate,
                channels=1,
                device=self.mic_device_id,
                callback=self.audio_callback,
                blocksize=int(self.mic_sample_rate * 0.1),  # 0.1 second chunks
                dtype='float32'
            )
            self.stream.start()
            
            # Start processing thread
            process_thread = threading.Thread(target=self._process_live_audio)
            process_thread.daemon = True
            process_thread.start()
            
        except Exception as e:
            self.stop_live_detection()
            messagebox.showerror("Stream Error", f"Failed to start audio stream:\n\n{str(e)}")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream - runs in separate thread"""
        if status:
            print(f"Audio status: {status}")
        
        # Add audio to queue
        self.audio_queue.put(indata.copy())
    
    def _process_live_audio(self):
        """Process live audio continuously"""
        while self.is_live_detecting:
            try:
                # Get audio from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Add to buffer
                self.live_buffer.extend(audio_chunk.flatten())
                
                # Keep buffer at fixed size (3 seconds)
                if len(self.live_buffer) > self.live_buffer_size:
                    self.live_buffer = self.live_buffer[-self.live_buffer_size:]
                
                # Process if we have enough data
                if len(self.live_buffer) >= self.live_buffer_size:
                    # Update visualization with current buffer
                    self.window.after(0, self.update_visualization, np.array(self.live_buffer))
                    
                    # Resample to 22050 Hz for feature extraction
                    audio_segment = np.array(self.live_buffer)
                    audio_resampled = librosa.resample(
                        audio_segment, 
                        orig_sr=self.mic_sample_rate, 
                        target_sr=self.sample_rate
                    )
                    
                    # Extract and store metrics
                    self.extract_and_store_metrics(audio_resampled)
                    self.window.after(0, self.update_audio_metrics_display)
                    
                    # Extract features
                    features = self.extract_features(audio_resampled)
                    
                    if features is not None:
                        # Predict
                        features_scaled = self.scaler.transform([features])
                        prob = self.model.predict_proba(features_scaled)[0, 1]
                        
                        # Add to predictions
                        self.live_predictions.append(prob)
                        
                        # Keep only last N predictions
                        if len(self.live_predictions) > self.live_prediction_window:
                            self.live_predictions = self.live_predictions[-self.live_prediction_window:]
                        
                        # Update UI with rolling average
                        mean_prob = np.mean(self.live_predictions)
                        self.window.after(0, self._update_live_results, mean_prob, len(self.live_predictions))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Live processing error: {e}")
                continue
    
    def _update_live_results(self, prob, num_samples):
        """Update UI with live detection results"""
        if not self.is_live_detecting:
            return
        
        # Use class threshold
        stenosis_threshold = 1.0 - (self.THRESHOLD / 100.0)
        prediction = "STENOSIS" if prob > stenosis_threshold else "NORMAL"
        
        # Calculate confidence based on distance from threshold
        normal_prob = 1 - prob
        confidence = self.calculate_threshold_confidence(normal_prob)
        
        # Color coding
        if prediction == 'STENOSIS':
            color = "#e74c3c"
            status_bg = '#e74c3c'
            status_text = "STENOSIS DETECTED - Continue monitoring"
        else:
            color = "#27ae60"
            status_bg = '#27ae60'
            status_text = "NORMAL - Continue monitoring"
        
        # Update display
        self.result_label.config(text=prediction, fg=color)
        
        self.segments_value.config(text=f"{num_samples} samples")
        self.duration_value.config(text="Live")
        self.confidence_score_value.config(text=f"{confidence:.1f}%")
        
        # Calculate consistency
        if len(self.live_predictions) > 1:
            std_prob = np.std(self.live_predictions)
            consistency = 1 - std_prob
            self.consistency_value.config(text=f"{consistency:.1%}")
        else:
            self.consistency_value.config(text="--")
        
        self.status_label.config(text=status_text, bg=status_bg)
    
    def stop_live_detection(self):
        """Stop live detection"""
        self.is_live_detecting = False
        
        # Stop stream
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        # Update UI
        self.live_detect_btn.config(
            text="START LIVE DETECTION",
            bg='#e74c3c',
            activebackground='#c0392b'
        )
        self.record_btn.config(state="normal")
        self.detect_btn.config(state="normal" if self.selected_file else "disabled")
        
        self.status_label.config(
            text="Live detection stopped - Ready for next operation",
            bg='#95a5a6'
        )
        
        # Show final summary if we have predictions
        if len(self.live_predictions) > 0:
            mean_prob = np.mean(self.live_predictions)
            stenosis_threshold = 1.0 - (self.THRESHOLD / 100.0)
            prediction = "STENOSIS" if mean_prob > stenosis_threshold else "NORMAL"
            self.info_label.config(
                text=f"Live Detection Summary:\n\nFinal Result: {prediction}\nAvg Probability: {mean_prob:.1%}\nSamples: {len(self.live_predictions)}\n\nReady for next operation."
            )
        else:
            self.info_label.config(text="Live detection stopped.\n\nReady for next operation.")
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording from microphone"""
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return
        
        if self.mic_device_id is None:
            messagebox.showerror("No Device", "No audio input device detected")
            return
        
        self.is_recording = True
        
        # Update UI
        self.record_btn.config(
            text="STOP RECORDING",
            bg='#95a5a6',
            activebackground='#7f8c8d'
        )
        self.live_detect_btn.config(state="disabled")
        self.detect_btn.config(state="disabled", bg='#95a5a6')
        self.clear_results()
        self.result_label.config(text="Recording...", fg="#9b59b6")
        self.status_label.config(
            text=f"Recording for {self.recording_duration} seconds...",
            bg='#9b59b6'
        )
        self.info_label.config(
            text=f"Recording in progress...\n\nDuration: {self.recording_duration}s\n\nPlace stethoscope on\nfistula site."
        )
        
        # Start recording in separate thread
        record_thread = threading.Thread(target=self._record_audio)
        record_thread.daemon = True
        record_thread.start()
    
    def _record_audio(self):
        """Record audio from microphone"""
        try:
            # Record audio at device's sample rate
            self.recording_data = sd.rec(
                int(self.recording_duration * self.mic_sample_rate),
                samplerate=self.mic_sample_rate,
                channels=1,
                device=self.mic_device_id,
                dtype='float32'
            )
            
            # Wait for recording to finish (with progress updates)
            for i in range(self.recording_duration):
                if not self.is_recording:
                    sd.stop()
                    return
                sd.sleep(1000)
                remaining = self.recording_duration - i - 1
                self.window.after(0, self._update_recording_progress, remaining)
            
            sd.wait()
            
            # Save recording to recordings folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recordings_dir = Path(__file__).parent / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            temp_file = recordings_dir / f"recording_{timestamp}.wav"
            sf.write(str(temp_file), self.recording_data, self.mic_sample_rate)
            
            # Set as selected file
            self.selected_file = str(temp_file)
            
            # Update UI - recording complete
            self.window.after(0, self._recording_complete, temp_file.name)
            
        except Exception as e:
            self.window.after(0, self._recording_error, str(e))
    
    def _update_recording_progress(self, remaining):
        """Update recording progress"""
        self.status_label.config(text=f"Recording... {remaining} seconds remaining")
    
    def _recording_complete(self, filename):
        """Handle recording completion"""
        self.is_recording = False
        
        self.record_btn.config(
            text="RECORD (30s)",
            bg='#9b59b6',
            activebackground='#8e44ad'
        )
        
        self.live_detect_btn.config(state="normal")
        
        self.file_label.config(
            text=f"Recorded: {filename}",
            fg='#2c3e50'
        )
        
        self.detect_btn.config(
            state="normal",
            bg='#27ae60',
            activebackground='#229954'
        )
        
        self.result_label.config(text="Recording Complete", fg="#27ae60")
        
        self.status_label.config(
            text="Recording saved - Click ANALYZE FILE to process",
            bg='#3498db'
        )
        
        self.info_label.config(
            text=f"Recording saved:\n{filename}\n\nReady to analyze.\nClick ANALYZE FILE button."
        )
    
    def _recording_error(self, error_msg):
        """Handle recording error"""
        self.is_recording = False
        
        self.record_btn.config(
            text="RECORD (30s)",
            bg='#9b59b6',
            activebackground='#8e44ad'
        )
        
        self.live_detect_btn.config(state="normal")
        
        self.result_label.config(text="ERROR", fg="#e74c3c")
        
        self.status_label.config(
            text=f"Recording error: {error_msg[:40]}",
            bg='#e74c3c'
        )
        
        messagebox.showerror("Recording Error", f"Failed to record:\n\n{error_msg}")
    
    def stop_recording(self):
        """Stop recording early"""
        self.is_recording = False
        sd.stop()
        
        self.record_btn.config(
            text="RECORD (30s)",
            bg='#9b59b6',
            activebackground='#8e44ad'
        )
        
        self.live_detect_btn.config(state="normal")
        
        self.status_label.config(
            text="Recording cancelled",
            bg='#95a5a6'
        )
    
    def select_file(self):
        """Open file dialog with card-based layout for easy touch selection"""
        # Create recordings directory if it doesn't exist
        recordings_dir = Path(__file__).parent / "recordings"
        recordings_dir.mkdir(exist_ok=True)
        
        # Create custom file dialog window
        dialog = tk.Toplevel(self.window)
        dialog.title("Select Recording")
        dialog.geometry("900x700")
        dialog.configure(bg='#2c3e50')
        
        # Make it modal
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (900 // 2)
        y = (dialog.winfo_screenheight() // 2) - (700 // 2)
        dialog.geometry(f"900x700+{x}+{y}")
        
        # Header
        header = tk.Label(
            dialog,
            text="Select Recording",
            font=("Arial", 16, "bold"),
            bg='#34495e',
            fg='white',
            pady=15
        )
        header.pack(fill='x')
        
        # Path label
        path_label = tk.Label(
            dialog,
            text=str(recordings_dir),
            font=("Arial", 10),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w',
            padx=10,
            pady=5
        )
        path_label.pack(fill='x', padx=20)
        
        # Scrollable canvas for cards
        canvas_frame = tk.Frame(dialog, bg='#2c3e50')
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        canvas = tk.Canvas(canvas_frame, bg='#2c3e50', highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2c3e50')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Load and display recordings as cards
        selected_file = [None]  # Use list to allow modification in nested function
        
        def select_file_and_close(filepath):
            selected_file[0] = filepath
            
            self.selected_file = str(filepath)
            self.file_label.config(
                text=f"{filepath.name}",
                fg='#2c3e50'
            )
            
            if self.model is not None:
                self.detect_btn.config(
                    state="normal",
                    bg='#27ae60',
                    activebackground='#229954'
                )
            
            self.clear_results()
            self.status_label.config(
                text=f"File selected: {filepath.name} - Click ANALYZE FILE",
                bg='#3498db'
            )
            self.info_label.config(
                text=f"Selected file:\n{filepath.name}\n\nReady to analyze.\nClick ANALYZE FILE button."
            )
            
            dialog.destroy()
        
        # Get all WAV files sorted by newest first
        try:
            wav_files = []
            for item in recordings_dir.iterdir():
                if item.suffix.lower() in ['.wav', '.wave']:
                    wav_files.append(item)
            
            wav_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(wav_files) == 0:
                # No recordings message
                no_files_label = tk.Label(
                    scrollable_frame,
                    text="No recordings found\n\nRecord audio first using the RECORD button",
                    font=("Arial", 14),
                    bg='#34495e',
                    fg='#ecf0f1',
                    pady=50
                )
                no_files_label.pack(fill='both', expand=True, padx=20, pady=20)
            else:
                # Create cards in grid (3 columns)
                row = 0
                col = 0
                
                for idx, filepath in enumerate(wav_files):
                    mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                    
                    # Card frame
                    card = tk.Frame(
                        scrollable_frame,
                        bg='#34495e',
                        relief='raised',
                        borderwidth=2,
                        cursor='hand2'
                    )
                    card.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
                    
                    # Make card clickable
                    def make_click_handler(fp):
                        return lambda e: select_file_and_close(fp)
                    
                    card.bind('<Button-1>', make_click_handler(filepath))
                    
                    # Audio icon (simple text badge)
                    icon_label = tk.Label(
                        card,
                        text="WAV",
                        font=("Arial", 14, "bold"),
                        bg='#3498db',
                        fg='white',
                        width=6,
                        height=2,
                        relief='raised',
                        borderwidth=2,
                        cursor='hand2'
                    )
                    icon_label.pack(pady=(8, 5))
                    icon_label.bind('<Button-1>', make_click_handler(filepath))
                    
                    # Recording number
                    number_label = tk.Label(
                        card,
                        text=f"Recording #{len(wav_files) - idx}",
                        font=("Arial", 11, "bold"),
                        bg='#34495e',
                        fg='white',
                        cursor='hand2'
                    )
                    number_label.pack(pady=(0, 3))
                    number_label.bind('<Button-1>', make_click_handler(filepath))
                    
                    # Date and time
                    datetime_label = tk.Label(
                        card,
                        text=mod_time.strftime("%Y-%m-%d\n%H:%M:%S"),
                        font=("Arial", 9),
                        bg='#34495e',
                        fg='#bdc3c7',
                        cursor='hand2'
                    )
                    datetime_label.pack(pady=(0, 10))
                    datetime_label.bind('<Button-1>', make_click_handler(filepath))
                    
                    # Configure grid column weights - 3 columns for better fit
                    scrollable_frame.grid_columnconfigure(col, weight=1, minsize=250)
                    
                    # Move to next position
                    col += 1
                    if col > 2:  # 3 columns
                        col = 0
                        row += 1
                        
        except Exception as e:
            error_label = tk.Label(
                scrollable_frame,
                text=f"Error loading recordings:\n{str(e)}",
                font=("Arial", 12),
                bg='#e74c3c',
                fg='white',
                pady=30
            )
            error_label.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Button frame
        button_frame = tk.Frame(dialog, bg='#2c3e50')
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Cancel button - centered
        cancel_btn = tk.Button(
            button_frame,
            text="CANCEL",
            command=dialog.destroy,
            font=("Arial", 14, "bold"),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            cursor="hand2",
            relief='flat',
            padx=50,
            pady=12
        )
        cancel_btn.pack(pady=10)
    
    def calculate_threshold_confidence(self, normal_prob):
        """
        Calculate confidence based on distance from threshold.
        Uses the class THRESHOLD setting.
        """
        actual_confidence = normal_prob * 100  # Convert to percentage
        
        if actual_confidence >= self.THRESHOLD:
            # For above threshold: confidence = 100 - (100 - actual_confidence)
            confidence = 100 - (100 - actual_confidence)
        else:
            # For below threshold: confidence = 100 - (THRESHOLD - actual_confidence)  
            confidence = 100 - (self.THRESHOLD - actual_confidence)
        
        return confidence
    
    def clear_results(self):
        """Clear previous results"""
        self.result_label.config(text="Ready", fg='#95a5a6')
        self.duration_value.config(text="--")
        self.segments_value.config(text="--")
        self.consistency_value.config(text="--")
        self.confidence_score_value.config(text="--")
        
        # Clear audio metrics
        self.audio_rms_value.config(text="--")
        self.audio_peak_value.config(text="--")
        self.audio_zcr_value.config(text="--")
        self.audio_dom_freq_value.config(text="--")
        self.audio_centroid_value.config(text="--")
        self.audio_rolloff_value.config(text="--")
        self.audio_bandwidth_value.config(text="--")
    
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
    
    def analyze_recording(self):
        """Analyze the selected recording"""
        if not self.selected_file:
            messagebox.showwarning("No File", "Please record or select a WAV file first")
            return
        
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return
        
        # Create stop event for timeout
        self.stop_analysis = threading.Event()
        
        # Run analysis in separate thread
        analysis_thread = threading.Thread(target=self._run_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # Start timeout monitor
        timeout_thread = threading.Thread(target=self._timeout_monitor, args=(analysis_thread, 60))
        timeout_thread.daemon = True
        timeout_thread.start()
    
    def _timeout_monitor(self, analysis_thread, timeout_seconds):
        """Monitor analysis thread and timeout if needed"""
        analysis_thread.join(timeout=timeout_seconds)
        
        if analysis_thread.is_alive():
            # Thread is still running after timeout
            self.stop_analysis.set()
            self.window.after(0, self._show_error, 
                            f"Analysis timeout after {timeout_seconds} seconds - File may be too long or corrupt")
    
    def _run_analysis(self):
        """Run the analysis (in separate thread)"""
        try:
            # Update UI - start
            self.window.after(0, self._analysis_started)
            
            # Load audio
            if self.stop_analysis.is_set():
                return
            audio, sr = librosa.load(self.selected_file, sr=self.sample_rate)
            duration = len(audio) / sr
            
            # Create segments
            if self.stop_analysis.is_set():
                return
            segments = self.create_segments(audio)
            
            # Extract features and predict
            predictions = []
            for i, segment in enumerate(segments):
                if self.stop_analysis.is_set():
                    return
                    
                features = self.extract_features(segment)
                if features is not None:
                    features_scaled = self.scaler.transform([features])
                    prob = self.model.predict_proba(features_scaled)[0, 1]
                    predictions.append(prob)
            
            # Extract metrics from first segment for display
            if len(segments) > 0:
                self.extract_and_store_metrics(segments[0])
                self.window.after(0, self.update_audio_metrics_display)
            
            if self.stop_analysis.is_set():
                return
                
            if len(predictions) == 0:
                self.window.after(0, self._show_error, "No valid features extracted")
                return
            
            # Aggregate - use class threshold
            mean_prob = np.mean(predictions)
            std_prob = np.std(predictions)
            # Use THRESHOLD setting
            stenosis_threshold = 1.0 - (self.THRESHOLD / 100.0)
            prediction = 1 if mean_prob > stenosis_threshold else 0
            
            result = {
                'prediction': 'STENOSIS' if prediction == 1 else 'NORMAL',
                'probability_stenosis': mean_prob,
                'probability_normal': 1 - mean_prob,
                'confidence': max(mean_prob, 1 - mean_prob),
                'num_segments': len(predictions),
                'duration': duration,
                'consistency': 1 - std_prob
            }
            
            if not self.stop_analysis.is_set():
                self.window.after(0, self._show_results, result)
            
        except Exception as e:
            if not self.stop_analysis.is_set():
                self.window.after(0, self._show_error, str(e))
    
    def _analysis_started(self):
        """Update UI when analysis starts"""
        self.detect_btn.config(state="disabled", bg='#95a5a6')
        self.record_btn.config(state="disabled")
        self.live_detect_btn.config(state="disabled")
        self.progress.start(10)
        self.result_label.config(text="Analyzing...", fg="#3498db")
        self.status_label.config(
            text="Processing audio segments...",
            bg='#f39c12'
        )
        self.info_label.config(
            text="Processing...\n\nExtracting features\nAnalyzing segments\nComputing results"
        )
    
    def _show_results(self, result):
        """Show analysis results"""
        self.progress.stop()
        self.detect_btn.config(
            state="normal",
            bg='#27ae60',
            activebackground='#229954'
        )
        self.record_btn.config(state="normal")
        self.live_detect_btn.config(state="normal")
        
        # Calculate threshold-based confidence
        normal_prob = result['probability_normal']
        confidence = self.calculate_threshold_confidence(normal_prob)
        
        # Color coding
        if result['prediction'] == 'STENOSIS':
            color = "#e74c3c"
            status_bg = '#e74c3c'
            status_text = "STENOSIS DETECTED - Please consult healthcare provider"
        else:
            color = "#27ae60"
            status_bg = '#27ae60'
            status_text = "NORMAL - No stenosis detected"
        
        # Update display
        self.result_label.config(
            text=result['prediction'],
            fg=color
        )
        
        self.duration_value.config(text=f"{result['duration']:.1f} seconds")
        self.segments_value.config(text=f"{result['num_segments']} segments")
        self.consistency_value.config(text=f"{result['consistency']:.1%}")
        self.confidence_score_value.config(text=f"{confidence:.1f}%")
        
        self.status_label.config(
            text=status_text,
            bg=status_bg
        )
        
        self.info_label.config(
            text=f"Analysis Complete!\n\nResult: {result['prediction']}\nConfidence: {confidence:.1f}%\n\nReady for next operation."
        )
    
    def _show_error(self, error_msg):
        """Show error message"""
        self.progress.stop()
        self.detect_btn.config(
            state="normal",
            bg='#27ae60',
            activebackground='#229954'
        )
        self.record_btn.config(state="normal")
        self.live_detect_btn.config(state="normal")
        self.result_label.config(text="ERROR", fg="#e74c3c")
        self.status_label.config(
            text=f"ERROR: {error_msg[:50]}",
            bg='#e74c3c'
        )
        self.info_label.config(
            text=f"ERROR:\n{error_msg}\n\nPlease try again."
        )
        messagebox.showerror("Analysis Error", f"Error analyzing file:\n\n{error_msg}")
    
    def run(self):
        """Start the application"""
        self.window.mainloop()


if __name__ == "__main__":
    app = StenosisTester()
    app.run()