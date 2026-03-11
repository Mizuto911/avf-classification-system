# live_screen.py
# -*- coding: utf-8 -*-

import tkinter as tk
import sounddevice as sd
import numpy as np
import librosa
import threading
import queue
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class LiveScreen:
    def __init__(self, parent, detector):
        self.parent = parent
        self.detector = detector
        
        # Audio settings
        self.mic_sample_rate = 48000
        self.mic_device_id = None
        
        # State
        self.is_detecting = False
        self.audio_queue = queue.Queue()
        self.live_buffer = []
        self.live_buffer_size = self.mic_sample_rate * 3
        self.live_predictions = []
        self.live_prediction_window = 10
        
        self.detect_usb_audio()
        self.setup_ui()
    
    def detect_usb_audio(self):
        """Detect USB audio device"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    if 'usb' in device_name or 'ab13x' in device_name:
                        self.mic_device_id = i
                        self.mic_sample_rate = int(device['default_samplerate'])
                        self.live_buffer_size = self.mic_sample_rate * 3
                        return
            # Use first available
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.mic_device_id = i
                    self.mic_sample_rate = int(device['default_samplerate'])
                    self.live_buffer_size = self.mic_sample_rate * 3
                    return
        except Exception as e:
            print(f"Error detecting audio: {e}")
    
    def setup_ui(self):
        """Setup UI"""
        # Title bar
        title_bar = tk.Frame(self.parent, bg='#7f8c8d', height=60)
        title_bar.pack(fill='x')
        title_bar.pack_propagate(False)
        
        tk.Label(
            title_bar,
            text="Real Time Monitoring",
            font=('Arial', 20, 'bold'),
            bg='#7f8c8d',
            fg='white'
        ).pack(pady=15)
        
        # Main content
        content = tk.Frame(self.parent, bg='#95a5a6')
        content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left - Visualization
        left = tk.Frame(content, bg='#95a5a6')
        left.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6), facecolor='#95a5a6')
        
        self.ax1 = self.fig.add_subplot(2, 1, 1)
        self.ax1.set_facecolor('#7f8c8d')
        self.ax1.set_title('Waveform', color='white', fontsize=11, fontweight='bold')
        self.ax1.tick_params(colors='white', labelsize=8)
        self.line1, = self.ax1.plot([], [], color='#3498db', linewidth=2)
        self.ax1.set_ylim(-1, 1)
        self.ax1.grid(True, alpha=0.2, color='white')
        for spine in self.ax1.spines.values():
            spine.set_color('white')
        
        self.ax2 = self.fig.add_subplot(2, 1, 2)
        self.ax2.set_facecolor('#7f8c8d')
        self.ax2.set_title('Frequency Spectrum', color='white', fontsize=11, fontweight='bold')
        self.ax2.tick_params(colors='white', labelsize=8)
        self.line2, = self.ax2.plot([], [], color='#e74c3c', linewidth=2)
        self.ax2.set_xlim(0, 2000)
        self.ax2.grid(True, alpha=0.2, color='white')
        for spine in self.ax2.spines.values():
            spine.set_color('white')
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Metrics below visualization
        metrics_panel = tk.Frame(left, bg='#7f8c8d', relief='solid', borderwidth=2)
        metrics_panel.pack(fill='x', pady=(10, 0))
        
        tk.Label(
            metrics_panel,
            text="LIVE AUDIO METRICS",
            font=('Arial', 10, 'bold'),
            bg='#7f8c8d',
            fg='white'
        ).pack(pady=8)
        
        metrics_grid = tk.Frame(metrics_panel, bg='#7f8c8d')
        metrics_grid.pack(padx=15, pady=(0, 10))
        
        # Create 2x4 grid for metrics
        self.rms_label = self.create_metric(metrics_grid, "RMS:", 0, 0)
        self.peak_label = self.create_metric(metrics_grid, "Peak:", 0, 2)
        self.zcr_label = self.create_metric(metrics_grid, "ZCR:", 1, 0)
        self.centroid_label = self.create_metric(metrics_grid, "Centroid:", 1, 2)
        
        # Right - Results
        right = tk.Frame(content, bg='#95a5a6', width=400)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)
        
        # Result card - RED style from image
        result_card = tk.Frame(right, bg='#c0392b', relief='raised', borderwidth=3)
        result_card.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            result_card,
            text="Analysis Results",
            font=('Arial', 14, 'bold'),
            bg='#c0392b',
            fg='#e8b4a8'
        ).pack(pady=(12, 5))
        
        self.status_label = tk.Label(
            result_card,
            text="Ready",
            font=('Arial', 36, 'bold'),
            bg='#c0392b',
            fg='white'
        )
        self.status_label.pack(pady=20)
        
        # Info rows
        info_container = tk.Frame(result_card, bg='#c0392b')
        info_container.pack(fill='x', padx=20, pady=(0, 15))
        
        self.create_info_row(info_container, "Duration:", "duration_label", 0)
        self.create_info_row(info_container, "Samples:", "samples_label", 1)
        self.create_info_row(info_container, "Confidence:", "confidence_label", 2)
        self.create_info_row(info_container, "Consistency:", "consistency_label", 3)
        
        # Button
        self.start_btn = tk.Button(
            right,
            text="Stop Live Detection",
            command=self.toggle_detection,
            bg='#c0392b',
            fg='white',
            font=('Arial', 14, 'bold'),
            relief='raised',
            borderwidth=3,
            cursor='hand2',
            pady=15
        )
        self.start_btn.pack(fill='x', pady=10)
        
        # Auto-start detection
        self.start_detection()
    
    def create_metric(self, parent, label, row, col):
        """Create metric display"""
        tk.Label(
            parent,
            text=label,
            font=('Arial', 9, 'bold'),
            bg='#7f8c8d',
            fg='white',
            width=10,
            anchor='w'
        ).grid(row=row, column=col, sticky='w', padx=5, pady=3)
        
        value = tk.Label(
            parent,
            text="--",
            font=('Arial', 9),
            bg='#7f8c8d',
            fg='white',
            anchor='w'
        )
        value.grid(row=row, column=col+1, sticky='w', padx=5, pady=3)
        return value
    
    def create_info_row(self, parent, label_text, attr_name, row):
        """Create info row in result card"""
        frame = tk.Frame(parent, bg='#c0392b')
        frame.pack(fill='x', pady=2)
        
        tk.Label(
            frame,
            text=label_text,
            font=('Arial', 11),
            bg='#c0392b',
            fg='white',
            width=12,
            anchor='w'
        ).pack(side='left')
        
        label = tk.Label(
            frame,
            text="Live" if "duration" in attr_name else "--",
            font=('Arial', 11),
            bg='#c0392b',
            fg='white',
            anchor='w'
        )
        label.pack(side='left')
        setattr(self, attr_name, label)
    
    def toggle_detection(self):
        """Toggle detection"""
        if self.is_detecting:
            self.stop_detection()
        else:
            self.start_detection()
    
    def start_detection(self):
        """Start detection"""
        if self.mic_device_id is None:
            self.status_label.config(text="No Device", fg='#e8b4a8')
            return
        
        self.is_detecting = True
        self.live_buffer = []
        self.live_predictions = []
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        self.start_btn.config(text="Stop Live Detection")
        self.status_label.config(text="Listening...", fg='white')
        
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
            threading.Thread(target=self.process_audio, daemon=True).start()
        except Exception as e:
            print(f"Error: {e}")
            self.stop_detection()
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback"""
        self.audio_queue.put(indata.copy())
    
    def process_audio(self):
        """Process audio"""
        while self.is_detecting:
            try:
                chunk = self.audio_queue.get(timeout=1.0)
                self.live_buffer.extend(chunk.flatten())
                
                if len(self.live_buffer) > self.live_buffer_size:
                    self.live_buffer = self.live_buffer[-self.live_buffer_size:]
                
                if len(self.live_buffer) >= self.live_buffer_size:
                    audio = np.array(self.live_buffer)
                    self.parent.after(0, self.update_visualization, audio)
                    
                    audio_resampled = librosa.resample(
                        audio,
                        orig_sr=self.mic_sample_rate,
                        target_sr=self.detector.sample_rate
                    )
                    
                    # Get metrics
                    metrics = self.detector.extract_audio_metrics(audio_resampled)
                    if metrics:
                        self.parent.after(0, self.update_metrics, metrics)
                    
                    # Predict
                    prob = self.detector.predict_segment(audio_resampled)
                    
                    if prob is not None:
                        self.live_predictions.append(prob)
                        if len(self.live_predictions) > self.live_prediction_window:
                            self.live_predictions = self.live_predictions[-self.live_prediction_window:]
                        
                        mean_prob = np.mean(self.live_predictions)
                        self.parent.after(0, self.update_results, mean_prob, len(self.live_predictions))
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Process error: {e}")
    
    def update_visualization(self, audio):
        """Update plots"""
        try:
            # Waveform
            waveform = audio[-1000:] if len(audio) > 1000 else audio
            x = np.arange(len(waveform))
            self.line1.set_data(x, waveform)
            self.ax1.set_xlim(0, len(waveform))
            
            # Spectrum
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1/self.mic_sample_rate)
            
            mask = freqs <= 2000
            self.line2.set_data(freqs[mask], magnitude[mask])
            max_mag = np.max(magnitude[mask])
            self.ax2.set_ylim(0, max_mag * 1.1 if max_mag > 0 else 1)
            
            self.canvas.draw_idle()
        except:
            pass
    
    def update_metrics(self, metrics):
        """Update metrics display"""
        self.rms_label.config(text=f"{metrics['rms']:.4f}")
        self.peak_label.config(text=f"{metrics['peak']:.4f}")
        self.zcr_label.config(text=f"{metrics['zcr']:.4f}")
        self.centroid_label.config(text=f"{metrics['spectral_centroid']:.0f} Hz")
    
    def update_results(self, prob, num_samples):
        """Update results"""
        stenosis_threshold = 1.0 - (self.detector.THRESHOLD / 100.0)
        prediction = "STENOSIS" if prob > stenosis_threshold else "NORMAL"
        
        normal_prob = 1 - prob
        confidence = self.detector.calculate_confidence(normal_prob)
        
        self.status_label.config(text=prediction)
        self.samples_label.config(text=f"{num_samples}")
        self.confidence_label.config(text=f"{confidence:.1f}%")
        
        if len(self.live_predictions) > 1:
            std_prob = np.std(self.live_predictions)
            consistency = 1 - std_prob
            self.consistency_label.config(text=f"{consistency:.1%}")
    
    def stop_detection(self):
        """Stop detection"""
        self.is_detecting = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        self.start_btn.config(text="Start Live Detection")
        self.status_label.config(text="Stopped", fg='#e8b4a8')
    
    def cleanup(self):
        """Cleanup"""
        self.stop_detection()