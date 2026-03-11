import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import librosa
import numpy as np
import joblib
from pathlib import Path
import threading
import warnings
warnings.filterwarnings('ignore')

class StenosisTester:
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
        
        self.setup_ui()
        self.load_model()
        
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
        
        # File selection section
        file_label_text = tk.Label(
            left_frame,
            text="Selected File:",
            font=("Arial", 11, "bold"),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        file_label_text.pack(anchor='w', pady=(0, 5))
        
        file_display_frame = tk.Frame(left_frame, bg='white', relief='solid', borderwidth=2)
        file_display_frame.pack(fill='x', pady=(0, 15))
        
        self.file_label = tk.Label(
            file_display_frame,
            text="No file selected",
            font=("Arial", 11),
            bg='white',
            fg='#7f8c8d',
            anchor='w',
            padx=10,
            pady=12,
            wraplength=280
        )
        self.file_label.pack(fill='both')
        
        # Browse button
        browse_btn = tk.Button(
            left_frame,
            text="BROWSE FILE",
            command=self.select_file,
            font=("Arial", 13, "bold"),
            bg='#3498db',
            fg='white',
            activebackground='#2980b9',
            activeforeground='white',
            cursor="hand2",
            relief='flat',
            borderwidth=0,
            height=2
        )
        browse_btn.pack(fill='x', pady=(0, 10))
        
        # Detect button
        self.detect_btn = tk.Button(
            left_frame,
            text="DETECT STENOSIS",
            command=self.analyze_recording,
            font=("Arial", 13, "bold"),
            bg='#95a5a6',
            fg='white',
            activebackground='#7f8c8d',
            activeforeground='white',
            cursor="hand2",
            state="disabled",
            relief='flat',
            borderwidth=0,
            height=2
        )
        self.detect_btn.pack(fill='x', pady=(0, 15))
        
        # Progress bar
        self.progress = ttk.Progressbar(
            left_frame,
            length=300,
            mode='indeterminate'
        )
        self.progress.pack(fill='x', pady=(0, 15))
        
        # Model status info
        status_info_frame = tk.Frame(left_frame, bg='#34495e', relief='solid', borderwidth=1)
        status_info_frame.pack(fill='both', expand=True)
        
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
            text="Ready to Analyze",
            font=("Arial", 32, "bold"),
            bg='#34495e',
            fg='#95a5a6'
        )
        self.result_label.pack(pady=(10, 10))
        
        # Confidence
        self.confidence_label = tk.Label(
            right_frame,
            text="",
            font=("Arial", 18, "bold"),
            bg='#34495e',
            fg='#ecf0f1'
        )
        self.confidence_label.pack(pady=(0, 8))
        
        # Separator line
        separator = tk.Frame(right_frame, bg='#7f8c8d', height=2)
        separator.pack(fill='x', padx=20, pady=10)
        
        # Details section
        details_container = tk.Frame(right_frame, bg='#34495e')
        details_container.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        
        # Duration row
        duration_frame = tk.Frame(details_container, bg='#34495e')
        duration_frame.pack(fill='x', pady=3)
        
        duration_label_text = tk.Label(
            duration_frame,
            text="Duration:",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=12,
            anchor='w'
        )
        duration_label_text.pack(side='left')
        
        self.duration_value = tk.Label(
            duration_frame,
            text="--",
            font=("Arial", 11),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.duration_value.pack(side='left', fill='x', expand=True)
        
        # Segments row
        segments_frame = tk.Frame(details_container, bg='#34495e')
        segments_frame.pack(fill='x', pady=3)
        
        segments_label_text = tk.Label(
            segments_frame,
            text="Segments:",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=12,
            anchor='w'
        )
        segments_label_text.pack(side='left')
        
        self.segments_value = tk.Label(
            segments_frame,
            text="--",
            font=("Arial", 11),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.segments_value.pack(side='left', fill='x', expand=True)
        
        # Consistency row
        consistency_frame = tk.Frame(details_container, bg='#34495e')
        consistency_frame.pack(fill='x', pady=3)
        
        consistency_label_text = tk.Label(
            consistency_frame,
            text="Consistency:",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=12,
            anchor='w'
        )
        consistency_label_text.pack(side='left')
        
        self.consistency_value = tk.Label(
            consistency_frame,
            text="--",
            font=("Arial", 11),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.consistency_value.pack(side='left', fill='x', expand=True)
        
        # Stenosis probability row
        stenosis_prob_frame = tk.Frame(details_container, bg='#34495e')
        stenosis_prob_frame.pack(fill='x', pady=3)
        
        stenosis_prob_label_text = tk.Label(
            stenosis_prob_frame,
            text="Stenosis Prob:",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=12,
            anchor='w'
        )
        stenosis_prob_label_text.pack(side='left')
        
        self.stenosis_prob_value = tk.Label(
            stenosis_prob_frame,
            text="--",
            font=("Arial", 11),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.stenosis_prob_value.pack(side='left', fill='x', expand=True)
        
        # Normal probability row
        normal_prob_frame = tk.Frame(details_container, bg='#34495e')
        normal_prob_frame.pack(fill='x', pady=3)
        
        normal_prob_label_text = tk.Label(
            normal_prob_frame,
            text="Normal Prob:",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='#bdc3c7',
            width=12,
            anchor='w'
        )
        normal_prob_label_text.pack(side='left')
        
        self.normal_prob_value = tk.Label(
            normal_prob_frame,
            text="--",
            font=("Arial", 11),
            bg='#34495e',
            fg='#ecf0f1',
            anchor='w'
        )
        self.normal_prob_value.pack(side='left', fill='x', expand=True)
        
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
        
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            # Get the directory where this script is located
            script_dir = Path(__file__).parent.absolute()
            model_path = script_dir / 'stenosis_model.pkl'
            scaler_path = script_dir / 'scaler.pkl'
            
            self.model = joblib.load(str(model_path))
            self.scaler = joblib.load(str(scaler_path))
            
            self.status_label.config(
                text="Model loaded successfully - Ready to analyze recordings",
                bg='#27ae60'
            )
            self.info_label.config(
                text="Model: Loaded\nScaler: Loaded\nStatus: Ready\n\nSelect a WAV file to begin analysis."
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
    
    def select_file(self):
        """Open file dialog to select WAV file"""
        filename = filedialog.askopenfilename(
            title="Select WAV File",
            filetypes=[
                ("WAV files", "*.wav *.WAV"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.selected_file = filename
            file_name = Path(filename).name
            
            # Update UI
            self.file_label.config(
                text=f"{file_name}",
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
                text=f"File selected: {file_name} - Click DETECT to analyze",
                bg='#3498db'
            )
            self.info_label.config(
                text=f"Selected file:\n{file_name}\n\nReady to analyze.\nClick DETECT button."
            )
    
    def clear_results(self):
        """Clear previous results"""
        self.result_label.config(text="Ready to Analyze", fg='#95a5a6')
        self.confidence_label.config(text="")
        self.duration_value.config(text="--")
        self.segments_value.config(text="--")
        self.consistency_value.config(text="--")
        self.stenosis_prob_value.config(text="--")
        self.normal_prob_value.config(text="--")
    
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
            messagebox.showwarning("No File", "Please select a WAV file first")
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
            
            if self.stop_analysis.is_set():
                return
                
            if len(predictions) == 0:
                self.window.after(0, self._show_error, "No valid features extracted")
                return
            
            # Aggregate
            mean_prob = np.mean(predictions)
            std_prob = np.std(predictions)
            prediction = 1 if mean_prob > 0.5 else 0
            
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
        self.progress.start(10)
        self.result_label.config(text="Analyzing...", fg="#3498db")
        self.confidence_label.config(text="Please wait")
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
        
        self.confidence_label.config(
            text=f"Confidence: {result['confidence']:.1%}",
            fg=color
        )
        
        self.duration_value.config(text=f"{result['duration']:.1f} seconds")
        self.segments_value.config(text=f"{result['num_segments']} segments")
        self.consistency_value.config(text=f"{result['consistency']:.1%}")
        self.stenosis_prob_value.config(text=f"{result['probability_stenosis']:.1%}")
        self.normal_prob_value.config(text=f"{result['probability_normal']:.1%}")
        
        self.status_label.config(
            text=status_text,
            bg=status_bg
        )
        
        self.info_label.config(
            text=f"Analysis Complete!\n\nResult: {result['prediction']}\nConfidence: {result['confidence']:.1%}\n\nReady for next file."
        )
    
    def _show_error(self, error_msg):
        """Show error message"""
        self.progress.stop()
        self.detect_btn.config(
            state="normal",
            bg='#27ae60',
            activebackground='#229954'
        )
        self.result_label.config(text="ERROR", fg="#e74c3c")
        self.confidence_label.config(text="Analysis Failed", fg="#e74c3c")
        self.status_label.config(
            text=f"ERROR: {error_msg[:50]}",
            bg='#e74c3c'
        )
        self.info_label.config(
            text=f"ERROR:\n{error_msg}\n\nPlease try again with a different file."
        )
        messagebox.showerror("Analysis Error", f"Error analyzing file:\n\n{error_msg}")
    
    def run(self):
        """Start the application"""
        self.window.mainloop()


if __name__ == "__main__":
    app = StenosisTester()
    app.run()