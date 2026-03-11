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
import time
import subprocess
warnings.filterwarnings('ignore')

class AVFDetectorApp:
    """Multi-screen AVF Stenosis Detector Application"""
    
    THRESHOLD = 90.0  # Classification threshold (% normal probability)
    
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
        
        # Recording parameters
        self.is_recording = False
        self.recording_data = None
        self.mic_device_id = None  # Will be auto-detected
        self.recording_duration = 30
        self.mic_sample_rate = 44100  # Standard sample rate
        
        # Live detection parameters
        self.is_live_detecting = False
        self.audio_queue = queue.Queue()
        self.live_buffer = []
        self.live_buffer_size = 44100 * 3  # 3 seconds at 44.1kHz
        self.live_predictions = []
        self.live_prediction_window = 10
        
        # Container for all screens
        self.screens = {}
        self.current_screen = None
        
        self.setup_ui()
        self.detect_microphone()
        self.load_model()
        self.show_screen("live_detection")
        
    def setup_ui(self):
        """Setup the main UI with sidebar and screen container"""
        # Main container
        main_container = tk.Frame(self.window, bg='#2c3e50')
        main_container.pack(fill='both', expand=True)
        
        # Sidebar (left side)
        self.sidebar = tk.Frame(main_container, bg='#2c3e50', width=140)
        self.sidebar.pack(side='left', fill='y')
        self.sidebar.pack_propagate(False)
        
        # Navigation buttons
        self.nav_buttons = []
        
        # Live Detection button
        live_btn_frame = tk.Frame(self.sidebar, bg='#2c3e50')
        live_btn_frame.pack(pady=(30, 20))
        
        live_btn = tk.Button(
            live_btn_frame,
            text="Live\nDetection",
            font=("Helvetica", 9),
            bg='#2c3e50',
            fg='white',
            activebackground='#34495e',
            bd=0,
            relief='flat',
            cursor="hand2",
            width=10,
            height=3,
            command=lambda: self.show_screen("live_detection")
        )
        live_btn.pack()
        self.nav_buttons.append(live_btn)
        
        # Recording button
        rec_btn_frame = tk.Frame(self.sidebar, bg='#2c3e50')
        rec_btn_frame.pack(pady=20)
        
        rec_btn = tk.Button(
            rec_btn_frame,
            text="Recording",
            font=("Helvetica", 9),
            bg='#2c3e50',
            fg='white',
            activebackground='#34495e',
            bd=0,
            relief='flat',
            cursor="hand2",
            width=10,
            height=3,
            command=lambda: self.show_screen("recording")
        )
        rec_btn.pack()
        self.nav_buttons.append(rec_btn)
        
        # File Scanner button
        file_btn_frame = tk.Frame(self.sidebar, bg='#2c3e50')
        file_btn_frame.pack(pady=20)
        
        file_btn = tk.Button(
            file_btn_frame,
            text="File\nScanner",
            font=("Helvetica", 9),
            bg='#2c3e50',
            fg='white',
            activebackground='#34495e',
            bd=0,
            relief='flat',
            cursor="hand2",
            width=10,
            height=3,
            command=lambda: self.show_screen("file_scanner")
        )
        file_btn.pack()
        self.nav_buttons.append(file_btn)
        
        # Shut Down button (bottom)
        shutdown_frame = tk.Frame(self.sidebar, bg='#2c3e50')
        shutdown_frame.pack(side='bottom', pady=30)
        
        shutdown_btn = tk.Button(
            shutdown_frame,
            text="Shut Down",
            font=("Helvetica", 8),
            bg='#2c3e50',
            fg='white',
            activebackground='#34495e',
            bd=0,
            relief='flat',
            cursor="hand2",
            width=10,
            height=2,
            command=self.confirm_shutdown
        )
        shutdown_btn.pack()
        
        # Right side container with screen and footer
        right_container = tk.Frame(main_container, bg='#ecf0f1')
        right_container.pack(side='right', fill='both', expand=True)
        
        # Screen container (content area)
        self.screen_container = tk.Frame(right_container, bg='#ecf0f1')
        self.screen_container.pack(fill='both', expand=True)
        
        # Footer at bottom
        footer = tk.Frame(right_container, bg='#2c3e50', height=40)
        footer.pack(side='bottom', fill='x')
        footer.pack_propagate(False)
        
        # Footer content
        footer_left = tk.Frame(footer, bg='#2c3e50')
        footer_left.pack(side='left', padx=20, pady=5)
        
        logo_label = tk.Label(
            footer_left,
            text="BRAVF ACOUSTIC ANALYTICS",
            font=("Helvetica", 8, "bold"),
            bg='#2c3e50',
            fg='#7f8c8d'
        )
        logo_label.pack()
        
        footer_right = tk.Frame(footer, bg='#2c3e50')
        footer_right.pack(side='right', padx=20, pady=5)
        
        copyright_label = tk.Label(
            footer_right,
            text="Copyright 2026 Mizunuma | Soriano | De Villa | Endaya. All Rights Reserved",
            font=("Helvetica", 7),
            bg='#2c3e50',
            fg='#5f6c7d'
        )
        copyright_label.pack()
        
        # Create all screens
        self.create_live_detection_screen()
        self.create_recording_screen()
        self.create_file_scanner_screen()
        
    def create_live_detection_screen(self):
        """Create the live detection screen"""
        screen = tk.Frame(self.screen_container, bg='white')
        self.screens["live_detection"] = screen
        
        header = tk.Frame(screen, bg='#6b7db5', height=50)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        header_label = tk.Label(
            header,
            text="Real Time Monitoring",
            font=("Helvetica", 16, "bold"),
            bg='#6b7db5',
            fg='white'
        )
        header_label.pack(expand=True, pady=10)
        
        content = tk.Frame(screen, bg='white')
        content.pack(fill='both', expand=True, padx=30, pady=15)
        
        result_container = tk.Frame(content, bg='#bdc3c7', bd=2, relief='solid')
        result_container.pack(fill='x', pady=(0, 15))
        
        self.live_result_frame = tk.Frame(result_container, bg='#95a5a6', height=90)
        self.live_result_frame.pack(fill='x', padx=2, pady=2)
        self.live_result_frame.pack_propagate(False)
        
        result_title = tk.Label(
            self.live_result_frame,
            text="Analysis Results",
            font=("Helvetica", 10, "bold"),
            bg='#95a5a6',
            fg='white'
        )
        result_title.pack(pady=(5, 2))
        
        self.live_result_label = tk.Label(
            self.live_result_frame,
            text="NOT STARTED",
            font=("Helvetica", 26, "bold"),
            bg='#95a5a6',
            fg='white'
        )
        self.live_result_label.pack(expand=True, pady=(0, 5))
        
        instructions_frame = tk.Frame(content, bg='#ecf0f1', bd=1, relief='solid')
        instructions_frame.pack(fill='x', pady=10)
        
        instructions = tk.Label(
            instructions_frame,
            text="Put the sensor on AVF\n\nApply the Stethoscope on AVF to start the analysis",
            font=("Helvetica", 10),
            bg='#ecf0f1',
            fg='#2c3e50',
            justify='center',
            pady=10
        )
        instructions.pack()
        
        self.live_control_btn = tk.Button(
            content,
            text="Start Live Detection",
            font=("Helvetica", 12, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            bd=0,
            command=self.toggle_live_detection,
            cursor="hand2",
            width=20,
            height=1,
            pady=8
        )
        self.live_control_btn.pack(pady=10)
        
    def create_recording_screen(self):
        """Create the recording screen"""
        screen = tk.Frame(self.screen_container, bg='white')
        self.screens["recording"] = screen
        
        header = tk.Frame(screen, bg='#6b7db5', height=50)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        header_label = tk.Label(
            header,
            text="Recording Session",
            font=("Helvetica", 16, "bold"),
            bg='#6b7db5',
            fg='white'
        )
        header_label.pack(expand=True, pady=10)
        
        content = tk.Frame(screen, bg='white')
        content.pack(fill='both', expand=True, padx=30, pady=15)
        
        status_container = tk.Frame(content, bg='#c0392b', bd=2, relief='solid')
        status_container.pack(fill='x', pady=(0, 12))
        
        self.rec_status_frame = tk.Frame(status_container, bg='#e74c3c', height=80)
        self.rec_status_frame.pack(fill='x', padx=2, pady=2)
        self.rec_status_frame.pack_propagate(False)
        
        status_title = tk.Label(
            self.rec_status_frame,
            text="Status",
            font=("Helvetica", 10, "bold"),
            bg='#e74c3c',
            fg='white'
        )
        status_title.pack(pady=(5, 2))
        
        self.rec_status_label = tk.Label(
            self.rec_status_frame,
            text="NOT RECORDING",
            font=("Helvetica", 22, "bold"),
            bg='#e74c3c',
            fg='#f5b7b1'
        )
        self.rec_status_label.pack(expand=True, pady=(0, 5))
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure("rec.Horizontal.TProgressbar",
                       troughcolor='#d5d8dc',
                       bordercolor='#d5d8dc',
                       background='#27ae60',
                       lightcolor='#27ae60',
                       darkcolor='#27ae60',
                       thickness=18)
        
        progress_frame = tk.Frame(content, bg='#bdc3c7', bd=1, relief='solid')
        progress_frame.pack(fill='x', pady=10)
        
        self.rec_progress = ttk.Progressbar(
            progress_frame,
            length=400,
            mode='determinate',
            style="rec.Horizontal.TProgressbar"
        )
        self.rec_progress.pack(padx=2, pady=2)
        
        instructions_frame = tk.Frame(content, bg='#ecf0f1', bd=1, relief='solid')
        instructions_frame.pack(fill='x', pady=10)
        
        self.rec_instructions = tk.Label(
            instructions_frame,
            text="Put the sensor on AVF\n\nA 30 second segment of your AVF acoustic vibration\nwill be recorded for analysis.",
            font=("Helvetica", 9),
            bg='#ecf0f1',
            fg='#2c3e50',
            justify='center',
            pady=8
        )
        self.rec_instructions.pack()
        
        self.rec_control_btn = tk.Button(
            content,
            text="Start Recording",
            font=("Helvetica", 12, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            bd=0,
            command=self.toggle_recording,
            cursor="hand2",
            width=20,
            height=1,
            pady=8
        )
        self.rec_control_btn.pack(pady=10)
        
    def create_file_scanner_screen(self):
        """Create the file scanner screen"""
        screen = tk.Frame(self.screen_container, bg='white')
        self.screens["file_scanner"] = screen
        
        header = tk.Frame(screen, bg='#6b7db5', height=50)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        header_label = tk.Label(
            header,
            text="File Scanning",
            font=("Helvetica", 16, "bold"),
            bg='#6b7db5',
            fg='white'
        )
        header_label.pack(expand=True, pady=10)
        
        content = tk.Frame(screen, bg='white')
        content.pack(fill='both', expand=True, padx=30, pady=12)
        
        # Result display
        result_container = tk.Frame(content, bg='#bdc3c7', bd=2, relief='solid')
        result_container.pack(fill='x', pady=(0, 20))
        
        self.file_result_frame = tk.Frame(result_container, bg='#95a5a6', height=90)
        self.file_result_frame.pack(fill='x', padx=2, pady=2)
        self.file_result_frame.pack_propagate(False)
        
        result_title = tk.Label(
            self.file_result_frame,
            text="Analysis Results",
            font=("Helvetica", 10, "bold"),
            bg='#95a5a6',
            fg='white'
        )
        result_title.pack(pady=(5, 2))
        
        self.file_result_label = tk.Label(
            self.file_result_frame,
            text="Ready",
            font=("Helvetica", 26, "bold"),
            bg='#95a5a6',
            fg='white'
        )
        self.file_result_label.pack(expand=True, pady=(0, 5))
        
        # Current Audio label
        audio_frame = tk.Frame(content, bg='#f8f9fa', bd=1, relief='solid')
        audio_frame.pack(fill='x', pady=5)
        
        self.audio_label = tk.Label(
            audio_frame,
            text="Current Audio: No file selected",
            font=("Helvetica", 8),
            bg='#f8f9fa',
            fg='#2c3e50',
            anchor='w',
            padx=8,
            pady=5
        )
        self.audio_label.pack(fill='x')
        
        # Buttons
        button_frame = tk.Frame(content, bg='white')
        button_frame.pack(fill='x', pady=15)
        
        select_btn = tk.Button(
            button_frame,
            text="Select Audio",
            font=("Helvetica", 11, "bold"),
            bg='#34495e',
            fg='white',
            activebackground='#2c3e50',
            bd=0,
            command=self.select_audio_file,
            cursor="hand2",
            width=14,
            height=1,
            pady=6
        )
        select_btn.pack(side='left', padx=(0, 8))
        
        self.analyze_btn = tk.Button(
            button_frame,
            text="Start Detection",
            font=("Helvetica", 11, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            bd=0,
            command=self.analyze_file,
            cursor="hand2",
            width=14,
            height=1,
            pady=6
        )
        self.analyze_btn.pack(side='right', padx=(8, 0))
        
    def show_screen(self, screen_name):
        """Show a specific screen"""
        if self.current_screen:
            self.screens[self.current_screen].pack_forget()
        
        self.screens[screen_name].pack(fill='both', expand=True)
        self.current_screen = screen_name
        
    def confirm_shutdown(self):
        """Show shutdown confirmation dialog"""
        dialog = tk.Toplevel(self.window)
        dialog.title("Shut Down")
        dialog.geometry("400x200")
        dialog.configure(bg='#ecf0f1')
        dialog.transient(self.window)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 200
        y = (dialog.winfo_screenheight() // 2) - 100
        dialog.geometry(f"400x200+{x}+{y}")
        
        tk.Label(
            dialog,
            text="Shut Down",
            font=("Helvetica", 16, "bold"),
            bg='#ecf0f1',
            fg='#2c3e50'
        ).pack(pady=(20, 10))
        
        tk.Label(
            dialog,
            text="Are you sure you want to\nshut down the device?",
            font=("Helvetica", 11),
            bg='#ecf0f1',
            fg='#2c3e50',
            justify='center'
        ).pack(pady=20)
        
        button_frame = tk.Frame(dialog, bg='#ecf0f1')
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame,
            text="Yes",
            font=("Helvetica", 12, "bold"),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            bd=0,
            command=lambda: self.shutdown_system(dialog),
            cursor="hand2",
            padx=30,
            pady=10
        ).pack(side='left', padx=10)
        
        tk.Button(
            button_frame,
            text="No",
            font=("Helvetica", 12, "bold"),
            bg='#e74c3c',
            fg='white',
            activebackground='#c0392b',
            bd=0,
            command=dialog.destroy,
            cursor="hand2",
            padx=30,
            pady=10
        ).pack(side='right', padx=10)

    def shutdown_system(self, dialog):
        """Destroy the app and shut down the Raspberry Pi"""
        dialog.destroy()
        self.window.destroy()
        subprocess.run(['sudo', 'shutdown', '-h', 'now'])
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            script_dir = Path(__file__).parent.absolute()
            model_path = script_dir / 'stenosis_model.pkl'
            scaler_path = script_dir / 'scaler.pkl'
            
            self.model = joblib.load(str(model_path))
            self.scaler = joblib.load(str(scaler_path))
            
        except FileNotFoundError:
            messagebox.showerror(
                "Model Not Found",
                "Cannot find model files in:\n" + str(Path(__file__).parent.absolute()) +
                "\n\nMake sure stenosis_model.pkl and scaler.pkl are present."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {str(e)}")
    
    def detect_microphone(self):
        """Detect and configure the microphone"""
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            
            if default_input:
                self.mic_device_id = default_input['index']
                
                supported_rates = [44100, 48000, 22050, 16000]
                for rate in supported_rates:
                    try:
                        sd.check_input_settings(
                            device=self.mic_device_id,
                            channels=1,
                            samplerate=rate
                        )
                        self.mic_sample_rate = rate
                        self.live_buffer_size = rate * 3
                        print(f"Microphone configured: Device {self.mic_device_id}, Rate: {rate}Hz")
                        break
                    except:
                        continue
            else:
                print("No input device found, recording/live detection may not work")
                
        except Exception as e:
            print(f"Error detecting microphone: {e}")
            self.mic_device_id = None
            self.mic_sample_rate = 44100
    
    # -- Live Detection ----------------------------------------------------------

    def toggle_live_detection(self):
        if not self.is_live_detecting:
            self.start_live_detection()
        else:
            self.stop_live_detection()
    
    def start_live_detection(self):
        if self.model is None:
            messagebox.showerror("No Model", "Model not loaded")
            return
        
        if self.mic_device_id is None:
            messagebox.showerror("No Microphone", "No microphone detected. Please connect a microphone and restart the application.")
            return
        
        self.is_live_detecting = True
        self.live_buffer = []
        self.live_predictions = []
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        self.live_control_btn.config(
            text="Stop Live Detection",
            bg='#e74c3c',
            activebackground='#c0392b'
        )
        
        self.live_result_frame.config(bg='#f39c12')
        self.live_result_label.config(text="Listening...", bg='#f39c12')
        
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
            
            process_thread = threading.Thread(target=self._process_live_audio)
            process_thread.daemon = True
            process_thread.start()
            
        except sd.PortAudioError as e:
            self.stop_live_detection()
            messagebox.showerror("Stream Error", f"Failed to start audio stream:\n\n{str(e)}\n\nPlease check your microphone connection.")
        except Exception as e:
            self.stop_live_detection()
            messagebox.showerror("Stream Error", f"Failed to start audio stream:\n\n{str(e)}")
    
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())
    
    def _process_live_audio(self):
        while self.is_live_detecting:
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
                self.live_buffer.extend(audio_chunk.flatten())
                
                if len(self.live_buffer) > self.live_buffer_size:
                    self.live_buffer = self.live_buffer[-self.live_buffer_size:]
                
                if len(self.live_buffer) >= self.live_buffer_size:
                    audio_segment = np.array(self.live_buffer)
                    audio_resampled = librosa.resample(
                        audio_segment, 
                        orig_sr=self.mic_sample_rate, 
                        target_sr=self.sample_rate
                    )
                    
                    features = self.extract_features(audio_resampled)
                    
                    if features is not None:
                        features_scaled = self.scaler.transform([features])
                        prob = self.model.predict_proba(features_scaled)[0, 1]
                        
                        self.live_predictions.append(prob)
                        
                        if len(self.live_predictions) > self.live_prediction_window:
                            self.live_predictions = self.live_predictions[-self.live_prediction_window:]
                        
                        mean_prob = np.mean(self.live_predictions)
                        self.window.after(0, self._update_live_results, mean_prob)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Live processing error: {e}")
                continue
    
    def _update_live_results(self, prob):
        if not self.is_live_detecting:
            return
        
        stenosis_threshold = 1.0 - (self.THRESHOLD / 100.0)
        prediction = "STENOSIS" if prob > stenosis_threshold else "NORMAL"
        
        if prediction == 'STENOSIS':
            self.live_result_frame.config(bg='#c0392b')
            self.live_result_label.config(text="STENOSIS", bg='#c0392b')
        else:
            self.live_result_frame.config(bg='#27ae60')
            self.live_result_label.config(text="NORMAL", bg='#27ae60')
    
    def stop_live_detection(self):
        self.is_live_detecting = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        self.live_control_btn.config(
            text="Start Live Detection",
            bg='#27ae60',
            activebackground='#229954'
        )
        
        self.live_result_frame.config(bg='#95a5a6')
        self.live_result_label.config(text="NOT STARTED", bg='#95a5a6')
    
    # -- Recording ---------------------------------------------------------------

    def toggle_recording(self):
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
        
        self.rec_control_btn.config(
            text="Stop Recording",
            bg='#e74c3c',
            activebackground='#c0392b'
        )
        
        self.rec_status_frame.config(bg='#27ae60')
        self.rec_status_label.config(text="RECORDING", bg='#27ae60', fg='white')
        
        self.rec_instructions.config(
            text="Recording Accoustic\nVibration\n\nA 30 second recording session\ncurrently in progress."
        )
        
        self.rec_progress['maximum'] = 100
        self.rec_progress['value'] = 0
        
        record_thread = threading.Thread(target=self._record_audio)
        record_thread.daemon = True
        record_thread.start()
    
    def _record_audio(self):
        try:
            self.recording_data = sd.rec(
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
                progress = ((i + 1) / self.recording_duration) * 100
                self.window.after(0, lambda p=progress: self.rec_progress.config(value=p))
            
            sd.wait()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recordings_dir = Path(__file__).parent / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            temp_file = recordings_dir / f"recording_{timestamp}.wav"
            sf.write(str(temp_file), self.recording_data, self.mic_sample_rate)
            
            self.selected_file = str(temp_file)
            self.window.after(0, self._recording_complete, temp_file.name)
            
        except sd.PortAudioError as e:
            error_msg = f"Audio device error: {str(e)}\n\nPlease check your microphone connection."
            self.window.after(0, self._recording_error, error_msg)
        except Exception as e:
            self.window.after(0, self._recording_error, str(e))
    
    def _recording_complete(self, filename):
        self.is_recording = False
        
        self.rec_control_btn.config(
            text="Start Recording",
            bg='#27ae60',
            activebackground='#229954'
        )
        
        self.rec_status_frame.config(bg='#e74c3c')
        self.rec_status_label.config(text="NOT RECORDING", bg='#e74c3c', fg='#f5b7b1')
        
        self.rec_progress['value'] = 100
        
        self.rec_instructions.config(
            text=f"Recording Complete!\n\n{filename}\n\nGo to File Scanner to analyze"
        )
    
    def _recording_error(self, error_msg):
        self.is_recording = False
        
        self.rec_control_btn.config(
            text="Start Recording",
            bg='#27ae60',
            activebackground='#229954'
        )
        
        self.rec_status_frame.config(bg='#e74c3c')
        self.rec_status_label.config(text="ERROR", bg='#e74c3c', fg='white')
        
        messagebox.showerror("Recording Error", f"Failed to record:\n\n{error_msg}")
    
    def stop_recording(self):
        self.is_recording = False
        sd.stop()
        
        self.rec_control_btn.config(
            text="Start Recording",
            bg='#27ae60',
            activebackground='#229954'
        )
        
        self.rec_status_frame.config(bg='#e74c3c')
        self.rec_status_label.config(text="NOT RECORDING", bg='#e74c3c', fg='#f5b7b1')
    
    # -- File Scanner ------------------------------------------------------------

    def select_audio_file(self):
        """Show file selection dialog"""
        recordings_dir = Path(__file__).parent / "recordings"
        recordings_dir.mkdir(exist_ok=True)
        
        dialog = tk.Toplevel(self.window)
        dialog.title("Recording List")
        dialog.geometry("600x500")
        dialog.configure(bg='white')
        dialog.transient(self.window)
        dialog.grab_set()
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 300
        y = (dialog.winfo_screenheight() // 2) - 250
        dialog.geometry(f"600x500+{x}+{y}")
        
        header_label = tk.Label(
            dialog,
            text="Recording List",
            font=("Helvetica", 16, "bold"),
            bg='white',
            fg='#2c3e50'
        )
        header_label.pack(pady=25)
        
        tk.Frame(dialog, bg='#bdc3c7', height=1).pack(fill='x', padx=40)
        
        canvas_frame = tk.Frame(dialog, bg='white')
        canvas_frame.pack(fill='both', expand=True, padx=40, pady=20)
        
        canvas = tk.Canvas(canvas_frame, bg='white', highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        try:
            wav_files = [f for f in recordings_dir.iterdir() if f.suffix.lower() in ['.wav', '.mp3']]
            wav_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(wav_files) == 0:
                tk.Label(
                    scrollable_frame,
                    text="No recordings found",
                    font=("Helvetica", 12),
                    bg='white',
                    fg='#7f8c8d'
                ).pack(pady=50)
            else:
                for filepath in wav_files:
                    file_label = tk.Label(
                        scrollable_frame,
                        text=filepath.name,
                        font=("Helvetica", 12),
                        bg='white',
                        fg='#2c3e50',
                        anchor='w',
                        padx=10,
                        pady=12,
                        cursor="hand2"
                    )
                    file_label.pack(fill='x', pady=2)
                    file_label.bind('<Button-1>', lambda e, f=filepath: self._select_file(f, dialog))
                    file_label.bind('<Enter>', lambda e, l=file_label: l.config(bg='#ecf0f1'))
                    file_label.bind('<Leave>', lambda e, l=file_label: l.config(bg='white'))
        except Exception as e:
            tk.Label(
                scrollable_frame,
                text=f"Error: {str(e)}",
                font=("Helvetica", 11),
                bg='white',
                fg='#e74c3c'
            ).pack(pady=50)
    
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
            
            predictions = []
            for segment in segments:
                features = self.extract_features(segment)
                if features is not None:
                    features_scaled = self.scaler.transform([features])
                    prob = self.model.predict_proba(features_scaled)[0, 1]
                    predictions.append(prob)
            
            if len(predictions) == 0:
                self.window.after(0, lambda: messagebox.showerror("Error", "No valid features extracted"))
                return
            
            mean_prob = np.mean(predictions)
            stenosis_threshold = 1.0 - (self.THRESHOLD / 100.0)
            prediction = 1 if mean_prob > stenosis_threshold else 0
            
            result = {
                'prediction': 'STENOSIS' if prediction == 1 else 'NORMAL',
            }
            
            self.window.after(0, self._show_file_results, result)
            
        except Exception as e:
            self.window.after(0, lambda: messagebox.showerror("Error", f"Analysis failed:\n{str(e)}"))
    
    def _show_file_results(self, result):
        if result['prediction'] == 'STENOSIS':
            self.file_result_frame.config(bg='#c0392b')
            self.file_result_label.config(text="STENOSIS", bg='#c0392b')
        else:
            self.file_result_frame.config(bg='#27ae60')
            self.file_result_label.config(text="NORMAL", bg='#27ae60')
        
        # Show completion dialog
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
        
        tk.Label(
            msg_dialog,
            text="Audio File Analysis",
            font=("Helvetica", 14, "bold"),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=(30, 20))
        
        tk.Label(
            msg_dialog,
            text="Acoustic Vibration Analysis\nfrom AVF Recording complete.",
            font=("Helvetica", 11),
            bg='white',
            fg='#2c3e50',
            justify='center'
        ).pack(pady=20)
        
        tk.Button(
            msg_dialog,
            text="OK",
            font=("Helvetica", 12, "bold"),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            bd=0,
            command=msg_dialog.destroy,
            cursor="hand2",
            padx=50,
            pady=12
        ).pack(pady=30)
    
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
    
    def create_segments(self, audio):
        seg_samples = int(self.segment_length * self.sample_rate)
        hop_samples = int(self.hop_length * self.sample_rate)
        
        segments = []
        for start in range(0, len(audio) - seg_samples, hop_samples):
            segment = audio[start:start + seg_samples]
            segments.append(segment)
        
        return segments
    
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = AVFDetectorApp()
    app.run()