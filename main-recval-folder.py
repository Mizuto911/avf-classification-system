# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import librosa
import numpy as np
import joblib
from pathlib import Path
import threading
import warnings
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import time
import psutil
import csv

warnings.filterwarnings('ignore')

class AVFDetectorApp:
    """Multi-screen AVF Stenosis Detector Application"""

    THRESHOLD = 90.0

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AVF Stenosis Detector")
        self.window.geometry("800x480") # Set a default size for better visibility
        self.window.configure(bg='#2c3e50')

        # Model parameters
        self.model = None
        self.scaler = None
        self.segment_length = 3
        self.hop_length = 2
        self.sample_rate = 22050
        self.cpu_monitor_interval = 0.1 # Faster polling for peak detection
        self.selected_file = None
        self.is_recording = False
        self.mic_sample_rate = 44100
        self.mic_device_id = None
        self.recording_duration = 30

        self.screens = {}
        self.current_screen = None
        
        # Performance Tracking
        self.peak_cpu_during_analysis = 0
        self.is_analyzing = False

        self.setup_ui()
        self.load_model()
        self.detect_microphone()
        self.show_screen("file_scanner")
        self.show_cpu_percent()

        
        audio_dir = Path(__file__).parent / "data" / "data_to_scan"
        self.audio_files = []
        self.output = []

        print('Scanning Files')

        try:
            self.audio_files = [f for f in audio_dir.iterdir() if f.suffix.lower() in ['.wav', '.mp3']]
            self.audio_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        except Exception as e:
            print(str(e))

        print(self.audio_files)

    def setup_ui(self):
        main_container = tk.Frame(self.window, bg='#2c3e50')
        main_container.pack(fill='both', expand=True)

        self.sidebar = tk.Frame(main_container, bg='#2c3e50', width=140)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)

        right_container = tk.Frame(main_container, bg='#ecf0f1')
        right_container.pack(side='right', fill='both', expand=True)

        self.screen_container = tk.Frame(right_container, bg='#ecf0f1')
        self.screen_container.pack(fill='both', expand=True)

        footer = tk.Frame(right_container, bg='#2c3e50', height=40)
        footer.pack(side='bottom', fill='x')
        
        tk.Label(footer, text="BRAVF ACOUSTIC ANALYTICS", font=("Helvetica", 8, "bold"),
                 bg='#2c3e50', fg='#7f8c8d').pack(side='left', padx=20, pady=5)

        self.create_file_scanner_screen()
        
        self.cpu_percentage_label = tk.Label(main_container, text='CPU Usage:\n0%', bg='green', fg='white', 
                                            padx=10, pady=10, font=('Arial', 12, 'bold'), justify=tk.LEFT)
        self.cpu_percentage_label.place(x=10, y=10)

    def monitor_cpu(self):
        psutil.cpu_percent(interval=None)
        while True:
            usage = psutil.cpu_percent(interval=None)
            
            # Track peak usage if analysis is running
            if self.is_analyzing:
                if usage > self.peak_cpu_during_analysis:
                    self.peak_cpu_during_analysis = usage
            
            self.cpu_percentage_label.config(text=f'CPU Usage:\n{usage}%')
            time.sleep(self.cpu_monitor_interval)

    def show_cpu_percent(self):
        threading.Thread(target=self.monitor_cpu, daemon=True).start()

    def create_file_scanner_screen(self):
        screen = tk.Frame(self.screen_container, bg='white')
        self.screens["file_scanner"] = screen

        header = tk.Frame(screen, bg='#6b7db5', height=50)
        header.pack(fill='x')
        tk.Label(header, text="File Scanning", font=("Helvetica", 16, "bold"),
                 bg='#6b7db5', fg='white').pack(expand=True, pady=10)

        content = tk.Frame(screen, bg='white')
        content.pack(fill='both', expand=True, padx=30, pady=12)

        # Result display
        result_container = tk.Frame(content, bg='#bdc3c7', bd=2, relief='solid')
        result_container.pack(fill='x', pady=(0, 10))

        self.file_result_frame = tk.Frame(result_container, bg='#95a5a6', height=70)
        self.file_result_frame.pack(fill='x', padx=2, pady=2)
        self.file_result_frame.pack_propagate(False)

        self.file_result_label = tk.Label(self.file_result_frame, text="Ready",
                                          font=("Helvetica", 22, "bold"), bg='#95a5a6', fg='white')
        self.file_result_label.pack(expand=True)

        # SCROLLABLE TEXT AREA
        tk.Label(content, text="Analysis Details:", font=("Helvetica", 10, "bold"), bg='white').pack(anchor='w')
        self.audio_text_area = scrolledtext.ScrolledText(content, height=3, font=("Consolas", 9), 
                                                        bg='#f8f9fa', state='disabled', relief='solid')
        self.audio_text_area.pack(fill='both', expand=True, pady=5)

        # Progress bar
        self.rec_progress = ttk.Progressbar(content, length=400, mode='determinate')
        self.rec_progress.pack(fill='x', pady=5)

        button_frame = tk.Frame(content, bg='white')
        button_frame.pack(fill='x', pady=10)

        self.analyze_btn = tk.Button(
            button_frame, text="Start Detection", font=("Helvetica", 11, "bold"),
            bg='#27ae60', fg='white', command=self.analyze_file, cursor="hand2", width=14
        )
        self.analyze_btn.pack(side='right')

    def show_screen(self, screen_name):
        if self.current_screen:
            self.screens[self.current_screen].pack_forget()
        self.screens[screen_name].pack(fill='both', expand=True)
        self.current_screen = screen_name

    def _start_analysis_phase(self):
        self.is_recording = False
        self.analyze_btn.config(text="Start Recording", bg='#27ae60', state=tk.DISABLED)
        self.analyze_file()

    def analyze_file(self):
        self.file_result_frame.config(bg='#f39c12')
        self.file_result_label.config(text="Analyzing...", bg='#f39c12')
        self.is_analyzing = True
        self.peak_cpu_during_analysis = 0 # Reset peak
        threading.Thread(target=self._run_file_analysis, daemon=True).start()

    def _run_file_analysis(self):
        try:
            for file in self.audio_files:
                self.is_analyzing = True
                self.peak_cpu_during_analysis = 0
                start_time = time.perf_counter()
                audio, sr = librosa.load(file, sr=self.sample_rate)
                segments = self.create_segments(audio)

                spectral_values = {"Spectral Centroid": [], "Spectral Roll-off": [], 
                                "Spectral Bandwidth": [], "Spectral Contrast": []}
                predictions = []

                for seg in segments:
                    features = self.extract_features(seg)
                    if features is None: continue

                    scaled = self.scaler.transform([features])
                    prob = self.model.predict_proba(scaled)[0, 1]
                    predictions.append(prob)

                    spectral_values["Spectral Centroid"].append(features[26])
                    spectral_values["Spectral Roll-off"].append(features[27])
                    spectral_values["Spectral Bandwidth"].append(features[28])
                    spectral_values["Spectral Contrast"].append(np.mean(features[29:36]))

                stats = {}
                for key in spectral_values:
                    vals = np.array(spectral_values[key])
                    stats[key] = {"mean": np.mean(vals), "std": np.std(vals), "min": np.min(vals), "max": np.max(vals)}

                mean_prob = np.mean(predictions)
                prediction = "STENOSIS" if mean_prob > 0.1 else "NORMAL"
                end_time = time.perf_counter()
                
                # Stop analysis tracking
                self.is_analyzing = False
                proc_time_ms = (end_time - start_time) * 1000

                current_file_peak = self.peak_cpu_during_analysis
                self.is_analyzing = False

                self.window.after(0, lambda: self._finalize_ui(prediction, stats, proc_time_ms, current_file_peak))

                # UI Update
            
            with open('data/output.csv', 'w', newline='', encoding='utf-8') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=self.output[0].keys())
                dict_writer.writeheader()
                dict_writer.writerows(self.output)

        except Exception as e:
            self.is_analyzing = False
            self.window.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            print()

    def _finalize_ui(self, prediction, stats, proc_time, cp):
        color = '#c0392b' if prediction == 'STENOSIS' else '#27ae60'
        self.file_result_frame.config(bg=color)
        self.file_result_label.config(text='DONE', bg='#27ae60')

        output = {}
        
        info_text = f"--- ANALYSIS RESULTS ---\n"
        info_text += f"Processing Time: {proc_time:.2f} ms\n"
        info_text += f"Max CPU Usage: {self.peak_cpu_during_analysis}%\n"
        info_text += "------------------------\n\n"
        
        for key, v in stats.items():
            info_text += f"{key}:\n"
            info_text += f"  Mean : {v['mean']:.2f}\n"
            info_text += f"  Std  : {v['std']:.2f}\n"
            info_text += f"  Range: {v['min']:.2f} - {v['max']:.2f}\n\n"
            output.update({
                key + '_Mean': f"{v['mean']:.2f}",
                key + '_Std': f"{v['std']:.2f}",
                key + '_Range': f"{v['min']:.2f} - {v['max']:.2f}",
            })

        output.update({
                'Peak-CPU Usage': cp,
                'Processing Time': proc_time,
                'Prediction': prediction,
            })
        
        self.output.append(output)
            
        self._update_scroll_text(info_text)
        self.analyze_btn.config(state=tk.NORMAL)

    def _update_scroll_text(self, text):
        self.audio_text_area.config(state='normal')
        self.audio_text_area.delete('1.0', tk.END)
        self.audio_text_area.insert(tk.END, text)
        self.audio_text_area.config(state='disabled')
        self.audio_text_area.see(tk.END)

    def extract_features(self, audio_segment):
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

            return np.concatenate([mfccs_mean, mfccs_std, [spectral_centroid], [spectral_rolloff], 
                                  [spectral_bandwidth], spectral_contrast_mean, [zcr], [rms]])
        except: return None

    def create_segments(self, audio):
        seg_samples = int(self.segment_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        return [audio[i:i + seg_samples] for i in range(0, len(audio) - seg_samples, hop_samples)]

    def detect_microphone(self):
        try:
            default_input = sd.query_devices(kind='input')
            if default_input:
                self.mic_device_id = default_input['index']
                self.mic_sample_rate = int(default_input['default_samplerate'])
        except: self.mic_device_id = None

    def load_model(self):
        try:
            self.model = joblib.load("stenosis_model.pkl")
            self.scaler = joblib.load("scaler.pkl")
        except: print("Model files missing.")

    def _recording_error(self, error_msg):
        self.is_recording = False
        self.analyze_btn.config(text="Start Recording", state=tk.NORMAL)
        messagebox.showerror("Error", error_msg)

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = AVFDetectorApp()
    app.run()




    