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

        stenosis_dir = Path(__file__).parent / "data" / "stenosis"
        stenosis_dir.mkdir(exist_ok=True)
        normal_dir = Path(__file__).parent / "data" / "normal"
        normal_dir.mkdir(exist_ok=True)

        self.stenosis_files = []
        self.normal_files = []
        self.stenosis_values = []
        self.normal_values = []

        try:
            self.stenosis_files = [f for f in stenosis_dir.iterdir() if f.suffix.lower() in ['.wav', '.mp3']]
            self.stenosis_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            print(f'\nStenosis:\n{self.stenosis_files}')
            self.normal_files = [f for f in normal_dir.iterdir() if f.suffix.lower() in ['.wav', '.mp3']]
            self.normal_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            print(f'\nNormal:\n{self.normal_files}')
        except Exception as e:
            print(str(e))
        
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
        tk.Label(header, text="Range Finder", font=("Helvetica", 16, "bold"),
                 bg='#6b7db5', fg='white').pack(expand=True, pady=10)
        
        content = tk.Frame(screen, bg='white')
        content.pack(fill='both', expand=True, padx=30, pady=12)
        
        result_container = tk.Frame(content, bg='#bdc3c7', bd=2, relief='solid')
        result_container.pack(fill='x', pady=(0, 20))

        self.stenosis_header_frame = tk.Frame(content, relief='solid')
        self.stenosis_header_frame.pack(fill='x')
        self.stenosis_header_label = tk.Label(self.stenosis_header_frame, text='STENOSIS',
                                    font=("Helvetica", 16), justify='left',
                                    anchor='w', padx=8, pady=5)
        self.stenosis_header_label.pack(fill='x')
        stenosis_frame = tk.Frame(content, bg='#f8f9fa', bd=1, relief='solid')
        stenosis_frame.pack(fill='x', pady=5)
        self.stenosis_label = tk.Label(stenosis_frame, text='Press analyze to show min and max values.',
                                    font=("Helvetica", 8), bg='#f8f9fa', fg='#2c3e50', justify='left',
                                    anchor='w', padx=8, pady=5)
        self.stenosis_label.pack(fill='x')

        self.normal_header_frame = tk.Frame(content, relief='solid')
        self.normal_header_frame.pack(fill='x')
        self.normal_header_label = tk.Label(self.normal_header_frame, text='NORMAL',
                                    font=("Helvetica", 16), justify='left',
                                    anchor='w', padx=8, pady=5)
        self.normal_header_label.pack(fill='x')
        normal_frame = tk.Frame(content, bg='#f8f9fa', bd=1, relief='solid')
        normal_frame.pack(fill='x', pady=5)
        self.normal_label = tk.Label(normal_frame, text='Press analyze to show min and max values.',
                                    font=("Helvetica", 8), bg='#f8f9fa', fg='#2c3e50', justify='left',
                                    anchor='w', padx=8, pady=5)
        self.normal_label.pack(fill='x')
        
        button_frame = tk.Frame(content, bg='white')
        button_frame.pack(fill='x', pady=15)
        
        self.analyze_all_btn = tk.Button(
            button_frame, text="Analyze All Files", font=("Helvetica", 11, "bold"),
            bg='#27ae60', fg='white', activebackground='#229954', bd=0,
            command=self.analyze_file, cursor="hand2", width=14, height=1, pady=6
        )
        self.analyze_all_btn.pack(side='right', padx=(8, 0))
        
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

    def analyze_file(self):
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return
        if not self.stenosis_files and not self.normal_files:
            messagebox.showerror("No Audio", "No Audio Files found")
            return
        
        analysis_thread = threading.Thread(target=self._run_file_analysis)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def _run_file_analysis(self):
        try:
            for file in self.stenosis_files:
                audio, sr = librosa.load(file, sr=self.sample_rate)
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
                for segment in segments:
                    features = self.extract_features(segment)
                    if features is not None:
                        i = 0
                        for key in keys:
                            average_values[key] = average_values[key] + features[i]
                            i = i + 1
                
                for key in keys:
                    average_values[key] = average_values[key] / len(segments)

                self.stenosis_values.append(average_values)

            for file in self.normal_files:
                audio, sr = librosa.load(file, sr=self.sample_rate)
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

                for segment in segments:
                    features = self.extract_features(segment)
                    if features is not None:
                        i = 0
                        for key in keys:
                            average_values[key] = average_values[key] + features[i]
                            i = i + 1
                
                for key in keys:
                    average_values[key] = average_values[key] / len(segments)

                self.normal_values.append(average_values)

            stenosis_value_range = self.get_min_max(self.stenosis_values)
            normal_value_range = self.get_min_max(self.normal_values)

            self.window.after(0, lambda: self.stenosis_label.config(text=self.get_min_max_content(stenosis_value_range)))
            self.window.after(0, lambda: self.normal_label.config(text=self.get_min_max_content(normal_value_range)))
            

        except Exception as e:
            self.window.after(0, lambda err=e: messagebox.showerror("Error", f"Analysis failed:\n{str(err)}"))
        
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
        return values_string
    
    def get_min_max(self, values: list):
        keys = values[0].keys()
        min_max_dict = {}
        for key in keys:
            value_list = list(value[key] for value in values)
            min_max_dict[key] = {
                'min': min(value_list),
                'max': max(value_list),
            }
        return min_max_dict
    
    def get_min_max_content(self, min_max_dict: dict):
        keys = min_max_dict.keys()
        min_max_content = ''
        for key in keys:
            min_max_content = min_max_content + f'{key}:\n Min - {min_max_dict[key]['min']}\nMax - {min_max_dict[key]['max']}\n'
        return min_max_content

    
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