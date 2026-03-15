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
        
        # Recording parameters
        self.is_recording = False
        self.recording_data = None
        self.mic_device_id = None
        self.recording_duration = 30
        self.mic_sample_rate = 44100
        
        # Live detection parameters
        self.is_live_detecting = False
        self.audio_queue = queue.Queue()
        self.live_buffer = []
        self.live_buffer_size = 44100 * 3
        self.live_predictions = []
        self.live_prediction_window = 10
        
        # Container for all screens
        self.screens = {}
        self.current_screen = None
        
        self.setup_ui()
        self.detect_microphone()
        self.load_model()
        self.show_screen("file_scanner")
        
    def setup_ui(self):
        """Setup the main UI with sidebar and screen container"""
        main_container = tk.Frame(self.window, bg='#2c3e50')
        main_container.pack(fill='both', expand=True)
        
        self.sidebar = tk.Frame(main_container, bg='#2c3e50', width=140)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)
        
        self.nav_buttons = []
        
        file_btn_frame = tk.Frame(self.sidebar, bg='#2c3e50')
        file_btn_frame.pack(pady=20)
        file_btn = tk.Button(
            file_btn_frame, text="File\nScanner", font=("Helvetica", 9),
            bg='#2c3e50', fg='white', activebackground='#34495e', bd=0,
            relief='flat', cursor="hand2", width=10, height=3,
            command=lambda: self.show_screen("file_scanner")
        )
        file_btn.pack()
        self.nav_buttons.append(file_btn)
        
        # Close Application button (bottom)
        shutdown_frame = tk.Frame(self.sidebar, bg='#2c3e50')
        shutdown_frame.pack(side='bottom', pady=30)
        shutdown_btn = tk.Button(
            shutdown_frame, text="Close App", font=("Helvetica", 8),
            bg='#2c3e50', fg='white', activebackground='#34495e', bd=0,
            relief='flat', cursor="hand2", width=10, height=2,
            command=self.confirm_close
        )
        shutdown_btn.pack()
        
        right_container = tk.Frame(main_container, bg='#ecf0f1')
        right_container.pack(side='right', fill='both', expand=True)
        
        self.screen_container = tk.Frame(right_container, bg='#ecf0f1')
        self.screen_container.pack(fill='both', expand=True)
        
        footer = tk.Frame(right_container, bg='#2c3e50', height=40)
        footer.pack(side='bottom', fill='x')
        footer.pack_propagate(False)
        
        footer_left = tk.Frame(footer, bg='#2c3e50')
        footer_left.pack(side='left', padx=20, pady=5)
        tk.Label(footer_left, text="BRAVF ACOUSTIC ANALYTICS", font=("Helvetica", 8, "bold"),
                 bg='#2c3e50', fg='#7f8c8d').pack()
        
        footer_right = tk.Frame(footer, bg='#2c3e50')
        footer_right.pack(side='right', padx=20, pady=5)
        tk.Label(footer_right,
                 text="Copyright 2026 Mizunuma | Soriano | De Villa | Endaya. All Rights Reserved",
                 font=("Helvetica", 7), bg='#2c3e50', fg='#5f6c7d').pack()
        
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
        
        tk.Label(self.file_result_frame, text="Analysis Results", font=("Helvetica", 10, "bold"),
                 bg='#95a5a6', fg='white').pack(pady=(5, 2))
        
        self.file_result_label = tk.Label(self.file_result_frame, text="Ready",
                                          font=("Helvetica", 26, "bold"), bg='#95a5a6', fg='white')
        self.file_result_label.pack(expand=True, pady=(0, 5))
        
        audio_frame = tk.Frame(content, bg='#f8f9fa', bd=1, relief='solid')
        audio_frame.pack(fill='x', pady=5)
        self.audio_label = tk.Label(audio_frame, text="Current Audio: No file selected",
                                    font=("Helvetica", 8), bg='#f8f9fa', fg='#2c3e50',
                                    anchor='w', padx=8, pady=5)
        self.audio_label.pack(fill='x')

        values_frame = tk.Frame(content, bg='#f8f9fa', bd=1, relief='solid')
        values_frame.pack(fill='x', pady=5)
        self.values_label = tk.Label(values_frame, text='No File Selected. No features to be shown.',
                                    font=("Helvetica", 8), bg='#f8f9fa', fg='#2c3e50',
                                    anchor='w', padx=8, pady=5)
        self.values_label.pack(fill='x')
        
        button_frame = tk.Frame(content, bg='white')
        button_frame.pack(fill='x', pady=15)
        
        tk.Button(button_frame, text="Stenosis", font=("Helvetica", 11, "bold"),
                  bg='#34495e', fg='white', activebackground='#2c3e50', bd=0,
                  command=lambda: self.select_audio_file("stenosis"), cursor="hand2", width=14, height=1,
                  pady=6).pack(side='left', padx=(0, 8))
        
        tk.Button(button_frame, text="Normal", font=("Helvetica", 11, "bold"),
                  bg='#34495e', fg='white', activebackground='#2c3e50', bd=0,
                  command=lambda: self.select_audio_file("normal"), cursor="hand2", width=14, height=1,
                  pady=6).pack(side='left', padx=(0, 8))
        
        self.analyze_btn = tk.Button(
            button_frame, text="Start Detection", font=("Helvetica", 11, "bold"),
            bg='#27ae60', fg='white', activebackground='#229954', bd=0,
            command=self.analyze_file, cursor="hand2", width=14, height=1, pady=6
        )
        self.analyze_btn.pack(side='right', padx=(8, 0))
        
    def show_screen(self, screen_name):
        if self.current_screen:
            self.screens[self.current_screen].pack_forget()
        self.screens[screen_name].pack(fill='both', expand=True)
        self.current_screen = screen_name
        
    def confirm_close(self):
        """Show close application confirmation dialog"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Close Application")
        dialog.geometry("400x200")
        dialog.configure(bg='#ecf0f1')
        dialog.transient(self.window)
        dialog.grab_set()
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 200
        y = (dialog.winfo_screenheight() // 2) - 100
        dialog.geometry(f"400x200+{x}+{y}")
        
        tk.Label(dialog, text="Close Application", font=("Helvetica", 16, "bold"),
                 bg='#ecf0f1', fg='#2c3e50').pack(pady=(20, 10))
        
        tk.Label(dialog, text="Are you sure you want to\nclose the application?",
                 font=("Helvetica", 11), bg='#ecf0f1', fg='#2c3e50', justify='center').pack(pady=20)
        
        button_frame = tk.Frame(dialog, bg='#ecf0f1')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Yes", font=("Helvetica", 12, "bold"),
                  bg='#27ae60', fg='white', activebackground='#229954', bd=0,
                  command=lambda: self.close_application(dialog),
                  cursor="hand2", padx=30, pady=10).pack(side='left', padx=10)
        
        tk.Button(button_frame, text="No", font=("Helvetica", 12, "bold"),
                  bg='#e74c3c', fg='white', activebackground='#c0392b', bd=0,
                  command=dialog.destroy, cursor="hand2", padx=30, pady=10).pack(side='right', padx=10)

    def close_application(self, dialog):
        """Close the application"""
        dialog.destroy()
        self.window.destroy()
        
    def load_model(self):
        try:
            script_dir = Path(__file__).parent.absolute()
            self.model = joblib.load(str(script_dir / 'stenosis_model.pkl'))
            self.scaler = joblib.load(str(script_dir / 'scaler.pkl'))
        except FileNotFoundError:
            messagebox.showerror("Model Not Found",
                "Cannot find model files in:\n" + str(Path(__file__).parent.absolute()) +
                "\n\nMake sure stenosis_model.pkl and scaler.pkl are present.")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
    
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
    
    # -- File Scanner ------------------------------------------------------------

    def select_audio_file(self, classification: str):
        file_dir = Path(__file__).parent / "data" / classification
        file_dir.mkdir(exist_ok=True)
        
        dialog = tk.Toplevel(self.window)
        dialog.title(f"{classification.capitalize()} List")
        dialog.geometry("600x500")
        dialog.configure(bg='white')
        dialog.transient(self.window)
        dialog.grab_set()
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 300
        y = (dialog.winfo_screenheight() // 2) - 250
        dialog.geometry(f"600x500+{x}+{y}")
        
        tk.Label(dialog, text=f"{classification.capitalize()} List", font=("Helvetica", 16, "bold"),
                 bg='white', fg='#2c3e50').pack(pady=25)
        tk.Frame(dialog, bg='#bdc3c7', height=1).pack(fill='x', padx=40)
        
        canvas_frame = tk.Frame(dialog, bg='white')
        canvas_frame.pack(fill='both', expand=True, padx=40, pady=20)
        
        canvas = tk.Canvas(canvas_frame, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        try:
            wav_files = [f for f in file_dir.iterdir() if f.suffix.lower() in ['.wav', '.mp3']]
            wav_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            if len(wav_files) == 0:
                tk.Label(scrollable_frame, text="No recordings found", font=("Helvetica", 12),
                         bg='white', fg='#7f8c8d').pack(pady=50)
            else:
                for filepath in wav_files:
                    file_label = tk.Label(scrollable_frame, text=filepath.name, font=("Helvetica", 12),
                                          bg='white', fg='#2c3e50', anchor='w', padx=10, pady=12, cursor="hand2")
                    file_label.pack(fill='x', pady=2)
                    file_label.bind('<Button-1>', lambda e, f=filepath: self._select_file(f, dialog))
                    file_label.bind('<Enter>', lambda e, l=file_label: l.config(bg='#ecf0f1'))
                    file_label.bind('<Leave>', lambda e, l=file_label: l.config(bg='white'))
        except Exception as e:
            tk.Label(scrollable_frame, text=f"Error: {str(e)}", font=("Helvetica", 11),
                     bg='white', fg='#e74c3c').pack(pady=50)
    
    def _select_file(self, filepath, dialog):
        self.selected_file = str(filepath)
        self.audio_label.config(text=f"Current Audio: {filepath.name}")
        dialog.destroy()
    
    def analyze_file(self):
        if not self.selected_file:
            messagebox.showwarning("No File", "Please select a recording first")
            return
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return
        
        self.file_result_frame.config(bg='#f39c12')
        self.file_result_label.config(text="Analyzing...", bg='#f39c12')
        
        analysis_thread = threading.Thread(target=self._run_file_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _run_file_analysis(self):
        try:
            audio, sr = librosa.load(self.selected_file, sr=self.sample_rate)
            segments = self.create_segments(audio)
            average_values = {
                "mfccs_mean": 0,
                "mfccs_std": 0,
                "spectral_centroid": 0,
                "spectral_rolloff": 0,
                "spectral_bandwidth": 0,
                "spectral_contrast_mean": 0,
                "zcr": 0,
                "rms": 0,
            }
            keys = average_values.keys()
            
            predictions = []
            for segment in segments:
                features = self.extract_features(segment)
                if features is not None:
                    features_scaled = self.scaler.transform([features])
                    prob = self.model.predict_proba(features_scaled)[0, 1]
                    predictions.append(prob)

                    for i in range(len(segments)):
                        average_values[keys[i]] = average_values[keys[i]] + features[i]
            
            if len(predictions) == 0:
                self.window.after(0, lambda: messagebox.showerror("Error", "No valid features extracted"))
                return
            
            for key in keys:
                average_values[key] = average_values[key] / len(segments)
            
            mean_prob = np.mean(predictions)
            stenosis_threshold = 1.0 - (self.THRESHOLD / 100.0)
            prediction = 'STENOSIS' if mean_prob > stenosis_threshold else 'NORMAL'
            self.window.after(0, self._show_file_results, {'prediction': prediction})
            self.values_label.config(text=self.get_values_string(average_values))
        except Exception as e:
            self.window.after(0, lambda: messagebox.showerror("Error", f"Analysis failed:\n{str(e)}"))
    
    def _show_file_results(self, result):
        if result['prediction'] == 'STENOSIS':
            self.file_result_frame.config(bg='#c0392b')
            self.file_result_label.config(text="STENOSIS", bg='#c0392b')
        else:
            self.file_result_frame.config(bg='#27ae60')
            self.file_result_label.config(text="NORMAL", bg='#27ae60')
        
        msg_dialog = tk.Toplevel(self.window)
        msg_dialog.title("Analysis Complete")
        msg_dialog.geometry("500x300")
        msg_dialog.configure(bg='white')
        msg_dialog.transient(self.window)
        msg_dialog.grab_set()
        
        msg_dialog.update_idletasks()
        x = (msg_dialog.winfo_screenwidth() // 2) - 250
        y = (msg_dialog.winfo_screenheight() // 2) - 150
        msg_dialog.geometry(f"500x300+{x}+{y}")
        
        tk.Label(msg_dialog, text="Audio File Analysis", font=("Helvetica", 14, "bold"),
                 bg='white', fg='#2c3e50').pack(pady=(30, 20))
        tk.Label(msg_dialog, text="Acoustic Vibration Analysis\nfrom AVF Recording complete.",
                 font=("Helvetica", 11), bg='white', fg='#2c3e50', justify='center').pack(pady=20)
        tk.Button(msg_dialog, text="OK", font=("Helvetica", 12, "bold"), bg='#3498db', fg='white',
                  activebackground='#2980b9', bd=0, command=msg_dialog.destroy,
                  cursor="hand2", padx=50, pady=12).pack(pady=30)
        
    def get_values_string(self, values: dict):
        if not self.selected_file:
            return 'No File Selected. No features to be shown.'
        values_string = ''
        keys = values.keys()
        for key in keys:
            values_string = values_string + f'{key}: {values[key]}\n'
        print(values_string)
        return values_string
    
    # -- Helpers -----------------------------------------------------------------

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
    
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = AVFDetectorApp()
    app.run()