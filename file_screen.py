# file_screen.py
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
import numpy as np
import librosa
import threading
from datetime import datetime
from pathlib import Path

class FileScreen:
    def __init__(self, parent, detector):
        self.parent = parent
        self.detector = detector
        
        self.selected_file = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        # Title bar
        title_bar = tk.Frame(self.parent, bg='#7f8c8d', height=60)
        title_bar.pack(fill='x')
        title_bar.pack_propagate(False)
        
        tk.Label(
            title_bar,
            text="File Scanning",
            font=('Arial', 20, 'bold'),
            bg='#7f8c8d',
            fg='white'
        ).pack(pady=15)
        
        # Center container
        container = tk.Frame(self.parent, bg='#95a5a6')
        container.place(relx=0.5, rely=0.5, anchor='center')
        
        # File display
        file_card = tk.Frame(container, bg='#7f8c8d', relief='solid', borderwidth=2, width=700)
        file_card.pack(pady=20)
        file_card.pack_propagate(False)
        
        self.file_label = tk.Label(
            file_card,
            text="No file selected",
            font=('Arial', 14),
            bg='#7f8c8d',
            fg='white',
            wraplength=650,
            pady=20
        )
        self.file_label.pack()
        
        # Buttons
        btn_frame = tk.Frame(container, bg='#95a5a6')
        btn_frame.pack(pady=15)
        
        tk.Button(
            btn_frame,
            text="BROWSE FILES",
            command=self.browse_files,
            bg='#2980b9',
            fg='white',
            font=('Arial', 14, 'bold'),
            width=18,
            relief='raised',
            borderwidth=3,
            cursor='hand2',
            pady=12
        ).pack(side='left', padx=5)
        
        self.analyze_btn = tk.Button(
            btn_frame,
            text="ANALYZE FILE",
            command=self.analyze_file,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 14, 'bold'),
            width=18,
            relief='raised',
            borderwidth=3,
            cursor='hand2',
            state='disabled',
            pady=12
        )
        self.analyze_btn.pack(side='left', padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(container, length=600, mode='indeterminate')
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
    
    def browse_files(self):
        """Browse files"""
        recordings_dir = Path(__file__).parent / "recordings"
        recordings_dir.mkdir(exist_ok=True)
        
        # Get main window
        main_window = self.parent.winfo_toplevel()
        
        dialog = tk.Toplevel(main_window)
        dialog.title("Select Recording")
        dialog.geometry("900x700")
        dialog.configure(bg='#2c3e50')
        dialog.transient(main_window)
        dialog.grab_set()
        
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 450
        y = (dialog.winfo_screenheight() // 2) - 350
        dialog.geometry(f"900x700+{x}+{y}")
        
        # Header
        header = tk.Frame(dialog, bg='#34495e')
        header.pack(fill='x')
        
        tk.Label(
            header,
            text="Select Recording",
            font=("Arial", 18, "bold"),
            bg='#34495e',
            fg='white',
            pady=20
        ).pack()
        
        tk.Label(
            dialog,
            text=str(recordings_dir),
            font=("Arial", 10),
            bg='#2c3e50',
            fg='#95a5a6',
            anchor='w',
            padx=20,
            pady=10
        ).pack(fill='x')
        
        # Scrollable canvas
        canvas_frame = tk.Frame(dialog, bg='#2c3e50')
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        canvas = tk.Canvas(canvas_frame, bg='#2c3e50', highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#2c3e50')
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Get files
        wav_files = sorted(
            [f for f in recordings_dir.iterdir() if f.suffix.lower() in ['.wav', '.wave']],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        def select_handler(filepath):
            self.selected_file = str(filepath)
            self.file_label.config(text=filepath.name)
            self.analyze_btn.config(state='normal', bg='#27ae60')
            dialog.destroy()
        
        if len(wav_files) == 0:
            tk.Label(
                scrollable_frame,
                text="No recordings found\n\nRecord audio first",
                font=("Arial", 16),
                bg='#2c3e50',
                fg='#95a5a6',
                pady=50
            ).pack(fill='both', expand=True)
        else:
            row = 0
            col = 0
            for idx, filepath in enumerate(wav_files):
                mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
                
                card = tk.Frame(scrollable_frame, bg='#34495e', relief='raised', borderwidth=2, cursor='hand2')
                card.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
                
                def make_handler(fp):
                    return lambda e: select_handler(fp)
                
                card.bind('<Button-1>', make_handler(filepath))
                
                # Icon
                icon = tk.Label(card, text="WAV", font=("Arial", 16, "bold"), bg='#3498db', fg='white',
                               width=6, height=2, relief='raised', borderwidth=1, cursor='hand2')
                icon.pack(pady=(10, 5))
                icon.bind('<Button-1>', make_handler(filepath))
                
                # Number
                number = tk.Label(card, text=f"Recording #{len(wav_files) - idx}", font=("Arial", 12, "bold"),
                                 bg='#34495e', fg='white', cursor='hand2')
                number.pack(pady=(0, 5))
                number.bind('<Button-1>', make_handler(filepath))
                
                # DateTime
                dt = tk.Label(card, text=mod_time.strftime("%Y-%m-%d\n%H:%M:%S"), font=("Arial", 10),
                             bg='#34495e', fg='#bdc3c7', cursor='hand2')
                dt.pack(pady=(0, 10))
                dt.bind('<Button-1>', make_handler(filepath))
                
                scrollable_frame.grid_columnconfigure(col, weight=1, minsize=280)
                
                col += 1
                if col > 2:
                    col = 0
                    row += 1
        
        # Cancel button
        tk.Button(
            dialog,
            text="CANCEL",
            command=dialog.destroy,
            bg='#95a5a6',
            fg='white',
            font=("Arial", 14, "bold"),
            cursor="hand2",
            relief='flat',
            padx=50,
            pady=12
        ).pack(pady=20)
    
    def analyze_file(self):
        """Analyze file"""
        if not self.selected_file:
            return
        
        self.analyze_btn.config(state='disabled', bg='#95a5a6')
        self.result_label.config(text="")
        self.result_details.config(text="")
        self.progress.start(10)
        
        threading.Thread(target=self.run_analysis, daemon=True).start()
    
    def run_analysis(self):
        """Run analysis"""
        try:
            audio, sr = librosa.load(self.selected_file, sr=self.detector.sample_rate)
            duration = len(audio) / sr
            
            segments = self.detector.create_segments(audio)
            predictions = []
            for segment in segments:
                prob = self.detector.predict_segment(segment)
                if prob is not None:
                    predictions.append(prob)
            
            if len(predictions) == 0:
                self.parent.after(0, self.show_error, "No valid features")
                return
            
            mean_prob = np.mean(predictions)
            std_prob = np.std(predictions)
            
            stenosis_threshold = 1.0 - (self.detector.THRESHOLD / 100.0)
            prediction = "STENOSIS" if mean_prob > stenosis_threshold else "NORMAL"
            
            normal_prob = 1 - mean_prob
            confidence = self.detector.calculate_confidence(normal_prob)
            consistency = 1 - std_prob
            
            self.parent.after(0, self.show_result, prediction, confidence, consistency, len(predictions), duration)
        
        except Exception as e:
            self.parent.after(0, self.show_error, str(e))
    
    def show_result(self, prediction, confidence, consistency, segments, duration):
        """Show result"""
        self.progress.stop()
        self.analyze_btn.config(state='normal', bg='#27ae60')
        
        self.result_label.config(text=prediction)
        
        details = f"Confidence: {confidence:.1f}%  |  Consistency: {consistency:.1%}\n"
        details += f"Duration: {duration:.1f}s  |  Segments: {segments}"
        self.result_details.config(text=details)
    
    def show_error(self, error):
        """Show error"""
        self.progress.stop()
        self.analyze_btn.config(state='normal', bg='#27ae60')
        self.result_label.config(text="ERROR")
        self.result_details.config(text=str(error))
    
    def cleanup(self):
        """Cleanup"""
        pass