# recording_screen.py
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import threading
from datetime import datetime
from pathlib import Path

class RecordingScreen:
    def __init__(self, parent, detector):
        self.parent = parent
        self.detector = detector
        
        # Settings
        self.mic_sample_rate = 48000
        self.mic_device_id = None
        self.recording_duration = 30
        self.is_recording = False
        
        self.detect_usb_audio()
        self.setup_ui()
    
    def detect_usb_audio(self):
        """Detect USB audio"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    if 'usb' in device_name or 'ab13x' in device_name:
                        self.mic_device_id = i
                        self.mic_sample_rate = int(device['default_samplerate'])
                        return
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.mic_device_id = i
                    self.mic_sample_rate = int(device['default_samplerate'])
                    return
        except:
            pass
    
    def setup_ui(self):
        """Setup UI"""
        # Title bar
        title_bar = tk.Frame(self.parent, bg='#7f8c8d', height=60)
        title_bar.pack(fill='x')
        title_bar.pack_propagate(False)
        
        tk.Label(
            title_bar,
            text="Recording Session",
            font=('Arial', 20, 'bold'),
            bg='#7f8c8d',
            fg='white'
        ).pack(pady=15)
        
        # Center container
        container = tk.Frame(self.parent, bg='#95a5a6')
        container.place(relx=0.5, rely=0.5, anchor='center')
        
        # Status
        self.status_label = tk.Label(
            container,
            text="Ready to record 30 seconds",
            font=('Arial', 18),
            bg='#95a5a6',
            fg='#2c3e50'
        )
        self.status_label.pack(pady=25)
        
        # Progress
        self.progress = ttk.Progressbar(container, length=600, mode='determinate')
        self.progress.pack(pady=20)
        
        # Result card
        result_card = tk.Frame(container, bg='#c0392b', relief='raised', borderwidth=3, width=700)
        result_card.pack(pady=30)
        result_card.pack_propagate(False)
        
        tk.Label(
            result_card,
            text="Analysis Results",
            font=('Arial', 14, 'bold'),
            bg='#c0392b',
            fg='#e8b4a8'
        ).pack(pady=(12, 5))
        
        self.result_label = tk.Label(
            result_card,
            text="",
            font=('Arial', 40, 'bold'),
            bg='#c0392b',
            fg='white'
        )
        self.result_label.pack(pady=20)
        
        self.result_details = tk.Label(
            result_card,
            text="",
            font=('Arial', 13),
            bg='#c0392b',
            fg='white',
            justify='center'
        )
        self.result_details.pack(pady=(0, 15), padx=25)
        
        # Button
        self.record_btn = tk.Button(
            container,
            text="START RECORDING",
            command=self.toggle_recording,
            bg='#8e44ad',
            fg='white',
            font=('Arial', 16, 'bold'),
            width=20,
            relief='raised',
            borderwidth=3,
            cursor='hand2',
            pady=15
        )
        self.record_btn.pack(pady=20)
    
    def toggle_recording(self):
        """Toggle recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording"""
        if self.mic_device_id is None:
            self.status_label.config(text="No audio device found")
            return
        
        self.is_recording = True
        self.record_btn.config(text="STOP", bg='#95a5a6', state='disabled')
        self.status_label.config(text="Recording...")
        self.result_label.config(text="")
        self.result_details.config(text="")
        threading.Thread(target=self.record_audio, daemon=True).start()
    
    def record_audio(self):
        """Record audio"""
        try:
            recording_data = sd.rec(
                int(self.recording_duration * self.mic_sample_rate),
                samplerate=self.mic_sample_rate,
                channels=1,
                device=self.mic_device_id,
                dtype='float32'
            )
            
            for i in range(self.recording_duration):
                if not self.is_recording:
                    sd.stop()
                    return
                sd.sleep(1000)
                progress = (i + 1) / self.recording_duration * 100
                remaining = self.recording_duration - i - 1
                self.parent.after(0, self.update_progress, progress, remaining)
            
            sd.wait()
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recordings_dir = Path(__file__).parent / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            filename = recordings_dir / f"recording_{timestamp}.wav"
            sf.write(str(filename), recording_data, self.mic_sample_rate)
            
            # Analyze
            audio = recording_data.flatten()
            audio_resampled = librosa.resample(
                audio,
                orig_sr=self.mic_sample_rate,
                target_sr=self.detector.sample_rate
            )
            
            segments = self.detector.create_segments(audio_resampled)
            predictions = []
            for segment in segments:
                prob = self.detector.predict_segment(segment)
                if prob is not None:
                    predictions.append(prob)
            
            if len(predictions) > 0:
                mean_prob = np.mean(predictions)
                std_prob = np.std(predictions)
                
                stenosis_threshold = 1.0 - (self.detector.THRESHOLD / 100.0)
                prediction = "STENOSIS" if mean_prob > stenosis_threshold else "NORMAL"
                
                normal_prob = 1 - mean_prob
                confidence = self.detector.calculate_confidence(normal_prob)
                consistency = 1 - std_prob
                
                self.parent.after(0, self.show_result, prediction, confidence, consistency, len(predictions), filename.name)
        
        except Exception as e:
            print(f"Error: {e}")
            self.parent.after(0, self.show_error, str(e))
    
    def update_progress(self, progress, remaining):
        """Update progress"""
        self.progress['value'] = progress
        self.status_label.config(text=f"Recording... {remaining}s remaining")
    
    def show_result(self, prediction, confidence, consistency, segments, filename):
        """Show result"""
        self.is_recording = False
        self.record_btn.config(text="START RECORDING", bg='#8e44ad', state='normal')
        self.status_label.config(text="Recording complete!")
        self.progress['value'] = 100
        
        self.result_label.config(text=prediction)
        
        details = f"Confidence: {confidence:.1f}%  |  Consistency: {consistency:.1%}\n"
        details += f"Segments: {segments}  |  Saved: {filename}"
        self.result_details.config(text=details)
    
    def show_error(self, error):
        """Show error"""
        self.is_recording = False
        self.record_btn.config(text="START RECORDING", bg='#8e44ad', state='normal')
        self.status_label.config(text=f"Error: {error}")
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        sd.stop()
        self.record_btn.config(text="START RECORDING", bg='#8e44ad', state='normal')
    
    def cleanup(self):
        """Cleanup"""
        self.stop_recording()