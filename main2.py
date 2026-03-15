# main.py
# -*- coding: utf-8 -*-

import tkinter as tk
from detection.detector import StenosisDetector
from live_screen import LiveScreen
from recording_screen import RecordingScreen
from file_screen import FileScreen

class StenosisApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AVF Stenosis Detector")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='#ecf0f1')
        
        # Load detector
        self.detector = StenosisDetector()
        
        # Current screen
        self.current_screen = None
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup main window with sidebar"""
        # Sidebar - Dark Navy
        sidebar = tk.Frame(self.root, bg='#1a2744', width=280)
        sidebar.pack(side='left', fill='y')
        sidebar.pack_propagate(False)
        
        # Back arrow at top
        back_frame = tk.Frame(sidebar, bg='#1a2744', height=80)
        back_frame.pack(fill='x')
        back_frame.pack_propagate(False)
        
        back_canvas = tk.Canvas(back_frame, width=40, height=40, bg='#1a2744', highlightthickness=0)
        back_canvas.pack(side='left', padx=25, pady=20)
        # Draw back arrow
        back_canvas.create_line(30, 20, 10, 20, fill='white', width=3, arrow=tk.FIRST, arrowshape=(12, 15, 5))
        
        # Navigation items
        nav_frame = tk.Frame(sidebar, bg='#1a2744')
        nav_frame.pack(fill='both', expand=True, pady=40)
        
        # Live Detection
        self.create_nav_item(nav_frame, "~", "Live Detection", self.show_live)
        
        # Recording
        self.create_nav_item(nav_frame, "|||", "Recording", self.show_recording)
        
        # File Scanner
        self.create_nav_item(nav_frame, "[]", "File Scanner", self.show_file)
        
        # Spacer
        tk.Frame(nav_frame, bg='#1a2744', height=80).pack()
        
        # Shut Down
        self.create_nav_item(nav_frame, "O", "Shut Down", self.root.quit, is_shutdown=True)
        
        # Logo at bottom
        logo_frame = tk.Frame(sidebar, bg='#1a2744')
        logo_frame.pack(side='bottom', fill='x', pady=25)
        
        # Circular logo
        logo_canvas = tk.Canvas(logo_frame, width=90, height=90, bg='#1a2744', highlightthickness=0)
        logo_canvas.pack(pady=10)
        logo_canvas.create_oval(10, 10, 80, 80, outline='white', width=3)
        logo_canvas.create_text(45, 45, text="BRAVF", fill='white', font=('Arial', 11, 'bold'))
        
        # Copyright text
        tk.Label(
            logo_frame,
            text="Copyright 2026 Mizunuma |\nSoriano | De Villa | Endaya.\nAll Rights Reserved",
            font=('Arial', 7),
            bg='#1a2744',
            fg='#8895a7',
            justify='center'
        ).pack()
        
        # Content area - Light gray
        self.content_frame = tk.Frame(self.root, bg='#95a5a6')
        self.content_frame.pack(side='right', fill='both', expand=True)
        
        # Show default screen
        self.show_live()
    
    def create_nav_item(self, parent, icon, text, command, is_shutdown=False):
        """Create navigation item with icon circle and text"""
        container = tk.Frame(parent, bg='#1a2744')
        container.pack(fill='x', padx=25, pady=12)
        
        # Icon circle container
        icon_container = tk.Frame(container, bg='#1a2744')
        icon_container.pack(side='left', padx=(0, 15))
        
        # Create circular canvas for icon
        circle_size = 55
        icon_canvas = tk.Canvas(
            icon_container,
            width=circle_size,
            height=circle_size,
            bg='#1a2744',
            highlightthickness=0,
            cursor='hand2'
        )
        icon_canvas.pack()
        
        # Draw circle
        circle_color = '#c0392b' if is_shutdown else '#3d5875'
        icon_canvas.create_oval(5, 5, circle_size-5, circle_size-5, fill=circle_color, outline='')
        
        # Draw icon text
        icon_canvas.create_text(
            circle_size/2, circle_size/2,
            text=icon,
            fill='white',
            font=('Arial', 18, 'bold')
        )
        
        icon_canvas.bind('<Button-1>', lambda e: command())
        
        # Text label
        text_label = tk.Label(
            container,
            text=text,
            font=('Arial', 13),
            bg='#1a2744',
            fg='white',
            cursor='hand2',
            anchor='w'
        )
        text_label.pack(side='left', fill='x', expand=True)
        text_label.bind('<Button-1>', lambda e: command())
    
    def clear_content(self):
        """Clear current screen"""
        if self.current_screen and hasattr(self.current_screen, 'cleanup'):
            self.current_screen.cleanup()
        
        for widget in self.content_frame.winfo_children():
            widget.destroy()
    
    def show_live(self):
        """Show live detection screen"""
        self.clear_content()
        self.current_screen = LiveScreen(self.content_frame, self.detector)
    
    def show_recording(self):
        """Show recording screen"""
        self.clear_content()
        self.current_screen = RecordingScreen(self.content_frame, self.detector)
    
    def show_file(self):
        """Show file scanning screen"""
        self.clear_content()
        self.current_screen = FileScreen(self.content_frame, self.detector)
    
    def run(self):
        """Run application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = StenosisApp()
    app.run()