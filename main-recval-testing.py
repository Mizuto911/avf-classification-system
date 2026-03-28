# -*- coding: utf-8 -*-
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
import soundfile as sf
from datetime import datetime
import queue
warnings.filterwarnings('ignore')


class AVFDetectorApp:
    """Multi-screen AVF Stenosis Detector Application"""

    THRESHOLD = 90.0  # Classification threshold (% normal probability)

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AVF Stenosis Detector")
        self.window.attributes('-fullscreen', False)
        self.window.configure(bg='#2c3e50')

        # Model parameters
        self.model = None
        self.scaler = None
        self.segment_length = 3
        self.hop_length = 2
        self.sample_rate = 22050
        self.selected_file = None
        self.is_recording = False
        self.mic_sample_rate = 44100
        self.mic_device_id = None
        self.recording_duration = 30
        self.is_recording = False

        # UI
        self.screens = {}
        self.current_screen = None

        self.setup_ui()
        self.load_model()
        self.detect_microphone()
        self.show_screen("file_scanner")

    # ================= UI =================
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
        footer.pack_propagate(False)

        tk.Label(footer, text="BRAVF ACOUSTIC ANALYTICS", font=("Helvetica", 8, "bold"),
                 bg='#2c3e50', fg='#7f8c8d').pack(side='left', padx=20, pady=5)

        tk.Label(footer, text="Copyright 2026 Mizunuma | Soriano | De Villa | Endaya. All Rights Reserved",
                 font=("Helvetica", 7), bg='#2c3e50', fg='#5f6c7d').pack(side='right', padx=20, pady=5)

        self.create_file_scanner_screen()

    def create_file_scanner_screen(self):
        screen = tk.Frame(self.screen_container, bg='white')
        self.screens["file_scanner"] = screen

        header = tk.Frame(screen, bg='#6b7db5', height=50)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="File Scanning", font=("Helvetica", 16, "bold"),
                 bg='#6b7db5', fg='white').pack(expand=True, pady=10)

        content = tk.Frame(screen, bg='white')
        content.pack(fill='both', expand=True, padx=30, pady=12)

        result_container = tk.Frame(content, bg='#bdc3c7', bd=2, relief='solid')
        result_container.pack(fill='x', pady=(0, 20))

        self.file_result_frame = tk.Frame(result_container, bg='#95a5a6', height=90)
        self.file_result_frame.pack(fill='x', padx=2, pady=2)
        self.file_result_frame.pack_propagate(False)

        tk.Label(self.file_result_frame, text="Testing Status", font=("Helvetica", 10, "bold"),
                 bg='#95a5a6', fg='white').pack(pady=(5, 2))

        self.file_result_label = tk.Label(self.file_result_frame, text="Ready",
                                          font=("Helvetica", 26, "bold"), bg='#95a5a6', fg='white')
        self.file_result_label.pack(expand=True, pady=(0, 5))

        progress_frame = tk.Frame(content, bg='#bdc3c7', bd=1, relief='solid')
        progress_frame.pack(fill='x', pady=10)
        self.rec_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate',
                                            style="rec.Horizontal.TProgressbar")
        self.rec_progress.pack(padx=2, pady=2)

        audio_frame = tk.Frame(content, bg='#f8f9fa', bd=1, relief='solid')
        audio_frame.pack(fill='x', pady=5)
        self.audio_label = tk.Label(audio_frame, text="Ready to Record and Analyze",
                                    font=("Helvetica", 8), bg='#f8f9fa', fg='#2c3e50',
                                    anchor='w', justify=tk.LEFT, padx=8, pady=5)
        self.audio_label.pack(fill='x')

        button_frame = tk.Frame(content, bg='white')
        button_frame.pack(fill='x', pady=15)

        self.analyze_btn = tk.Button(
            button_frame, text="Start Detection", font=("Helvetica", 11, "bold"),
            bg='#27ae60', fg='white', activebackground='#229954', bd=0,
            command=self.record_analyze_file, cursor="hand2", width=14, height=1, pady=6
        )
        self.analyze_btn.pack(side='right', padx=(8, 0))

    def show_screen(self, screen_name):
        if self.current_screen:
            self.screens[self.current_screen].pack_forget()
        self.screens[screen_name].pack(fill='both', expand=True)
        self.current_screen = screen_name

    # ================= RECORDING =================
    def record_analyze_file(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return
        if self.mic_device_id is None:
            messagebox.showerror("No Microphone", "No microphone detected. Please connect a microphone and restart the application.")
            return
        
        self.is_recording = True
        self.analyze_btn.config(text="Stop Recording", bg='#e74c3c', activebackground='#c0392b')
        self.file_result_frame.config(bg='#27ae60')
        self.file_result_label.config(text="RECORDING", bg='#27ae60', fg='white')
        self.audio_label.config(text="Recording Accoustic\nVibration\n\nA 30 second recording session\ncurrently in progress.")
        self.rec_progress['maximum'] = 100
        self.rec_progress['value'] = 0
        
        record_thread = threading.Thread(target=self._record_audio)
        record_thread.daemon = True
        record_thread.start()
    
    def _record_audio(self):
        try:
            self.recording_data = sd.rec(
                int(self.recording_duration * self.mic_sample_rate),
                samplerate=self.mic_sample_rate, channels=1,
                device=self.mic_device_id, dtype='float32'
            )
            for i in range(self.recording_duration):
                if not self.is_recording:
                    sd.stop()
                    return
                sd.sleep(1000)
                progress = ((i + 1) / self.recording_duration) * 100
                self.window.after(0, lambda p=progress: self.rec_progress.config(value=p))
            sd.wait()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recordings_dir = Path(__file__).parent / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            temp_file = recordings_dir / f"recording_{timestamp}.wav"
            sf.write(str(temp_file), self.recording_data, self.mic_sample_rate)
            self.selected_file = str(temp_file)
            self._recording_complete()
            self.analyze_btn.config(state=tk.DISABLED)
            self.analyze_file()
            self.analyze_btn.config(state=tk.NORMAL)
        except sd.PortAudioError as e:
            self.window.after(0, self._recording_error, f"Audio device error: {str(e)}\n\nPlease check your microphone connection.")
        except Exception as e:
            self.window.after(0, self._recording_error, str(e))
    
    def _recording_complete(self):
        self.is_recording = False
        self.analyze_btn.config(text="Start Recording", bg='#27ae60', activebackground='#229954')
        self.rec_progress['value'] = 100
    
    def _recording_error(self, error_msg):
        self.is_recording = False
        self.analyze_btn.config(text="Start Recording", bg='#27ae60', activebackground='#229954')
        self.file_result_frame.config(bg='#e74c3c')
        self.file_result_label.config(text="ERROR", bg='#e74c3c', fg='white')
        messagebox.showerror("Recording Error", f"Failed to record:\n\n{error_msg}")
    
    def stop_recording(self):
        self.is_recording = False
        sd.stop()
        self.analyze_btn.config(text="Start Recording", bg='#27ae60', activebackground='#229954')
        self.file_result_frame.config(bg='#e74c3c')
        self.file_result_label.config(text="NOT RECORDING", bg='#e74c3c', fg='#f5b7b1')

    # ================= ANALYSIS =================
    def analyze_file(self):
        if not self.selected_file:
            messagebox.showwarning("No File", "Please select a recording first")
            return
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return

        self.file_result_frame.config(bg='#f39c12')
        self.file_result_label.config(text="Analyzing...", bg='#f39c12')
        threading.Thread(target=self._run_file_analysis).start()

    def _run_file_analysis(self):
        try:
            audio, sr = librosa.load(self.selected_file, sr=self.sample_rate)
            segments = self.create_segments(audio)

            # Only keep 4 spectral features for UI display
            spectral_values = {
                "Spectral Centroid": [],
                "Spectral Roll-off": [],
                "Spectral Bandwidth": [],
                "Spectral Contrast": [],
            }

            predictions = []

            for seg in segments:
                features = self.extract_features(seg)
                if features is None:
                    continue

                scaled = self.scaler.transform([features])
                prob = self.model.predict_proba(scaled)[0, 1]
                predictions.append(prob)

                # Correct indices for spectral features
                spectral_values["Spectral Centroid"].append(features[26])
                spectral_values["Spectral Roll-off"].append(features[27])
                spectral_values["Spectral Bandwidth"].append(features[28])
                spectral_values["Spectral Contrast"].append(np.mean(features[29:36]))

            # Compute mean, std, min/max for UI
            stats = {}
            for key in spectral_values:
                vals = np.array(spectral_values[key])
                stats[key] = {
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals)
                }

            mean_prob = np.mean(predictions)
            prediction = "STENOSIS" if mean_prob > 0.1 else "NORMAL"

            self.window.after(0, lambda: self.file_result_frame.config(
                bg='#c0392b' if prediction == 'STENOSIS' else '#27ae60'))
            self.window.after(0, lambda: self.file_result_label.config(
                text=prediction,
                bg='#c0392b' if prediction == 'STENOSIS' else '#27ae60'))
            self.window.after(0, lambda: self.audio_label.config(text=self.get_values_string(stats)))

        except Exception as e:
            print(e)
            messagebox.showerror("Error", f"Analysis failed: {e}")

    def get_values_string(self, values: dict):
        # Only shows the 4 spectral features on UI
        text = ""
        for key, v in values.items():
            text += (
                f"{key}\n"
                f" Mean : {v['mean']:.2f}\n"
                f" Std  : {v['std']:.2f}\n"
                f" Range: {v['min']:.2f} - {v['max']:.2f}\n\n"
            )
        return text

    # ================= FEATURES =================
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

            return np.concatenate([
                mfccs_mean, mfccs_std,
                [spectral_centroid], [spectral_rolloff], [spectral_bandwidth],
                spectral_contrast_mean,
                [zcr], [rms]
            ])
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def create_segments(self, audio):
        seg_samples = int(self.segment_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        segments = []
        for start in range(0, len(audio) - seg_samples, hop_samples):
            segments.append(audio[start:start + seg_samples])
        return segments
    
    def detect_microphone(self):
        try:
            default_input = sd.query_devices(kind='input')
            if default_input:
                self.mic_device_id = default_input['index']
                for rate in [44100, 48000, 22050, 16000]:
                    try:
                        sd.check_input_settings(device=self.mic_device_id, channels=1, samplerate=rate)
                        self.mic_sample_rate = rate
                        self.live_buffer_size = rate * 3
                        print(f"Microphone configured: Device {self.mic_device_id}, Rate: {rate}Hz")
                        break
                    except:
                        continue
            else:
                print("No input device found")
        except Exception as e:
            print(f"Error detecting microphone: {e}")
            self.mic_device_id = None
            self.mic_sample_rate = 44100

    # ================= MODEL =================
    def load_model(self):
        try:
            self.model = joblib.load("stenosis_model.pkl")
            self.scaler = joblib.load("scaler.pkl")
        except:
            print("Model not found")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = AVFDetectorApp()
    app.run()