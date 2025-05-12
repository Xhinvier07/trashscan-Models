import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk
import torch
from ultralytics import YOLO
import numpy as np
import threading
from queue import Queue
import time
import os
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


class ModernSAHIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAHI YOLO Litter Detection")
        self.root.configure(bg='#f0f0f0')

        # Set minimum window size
        self.root.minsize(1000, 700)

        # Configure grid weight
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Initialize variables
        self.video_source = None
        self.is_video_playing = False
        self.video_thread = None
        self.frame_queue = Queue(maxsize=32)
        self.current_file = None
        self.is_video = False
        self.is_webcam = False
        
        # SAHI specific variables
        self.sahi_enabled = tk.BooleanVar(value=False)
        self.slice_size_var = tk.StringVar(value="1024")
        self.slice_overlap_var = tk.DoubleVar(value=0.2)
        self.sahi_detection_model = None
        self.yolo_model = None
        
        # Create and apply style
        self.create_style()

        # Setup UI
        self.setup_ui()

        # Load model automatically
        self.load_model()

    def create_style(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Configure colors
        style.configure('Main.TFrame', background='#f0f0f0')
        style.configure('Controls.TFrame', background='#ffffff', relief='flat')

        # Button styles
        style.configure('Primary.TButton',
                        padding=10,
                        font=('Helvetica', 10))

        style.configure('Secondary.TButton',
                        padding=8,
                        font=('Helvetica', 10))

        # Toggle button styles
        style.configure('Toggle.TButton',
                        padding=8,
                        font=('Helvetica', 10))
        style.configure('Toggle.On.TButton',
                        background='#4CAF50',
                        foreground='white')

        # Label styles
        style.configure('Status.TLabel',
                        background='#ffffff',
                        font=('Helvetica', 10))

        style.configure('Title.TLabel',
                        background='#f0f0f0',
                        font=('Helvetica', 12, 'bold'))

    def setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, style='Main.TFrame', padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Top controls frame
        self.setup_controls()

        # Canvas frame
        self.setup_canvas()

        # Status bar
        self.setup_status_bar()

    def setup_controls(self):
        controls_frame = ttk.Frame(self.main_frame, style='Controls.TFrame', padding="10")
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls_frame.grid_columnconfigure(6, weight=1)

        # Input controls
        ttk.Button(controls_frame,
                   text="Open Image",
                   style='Primary.TButton',
                   command=self.load_image).grid(row=0, column=0, padx=5)

        ttk.Button(controls_frame,
                   text="Open Video",
                   style='Primary.TButton',
                   command=self.load_video).grid(row=0, column=1, padx=5)

        # Confidence threshold
        ttk.Label(controls_frame,
                  text="Confidence:",
                  style='Status.TLabel').grid(row=0, column=2, padx=(20, 5))

        self.conf_threshold = ttk.Scale(controls_frame,
                                        from_=0.0,
                                        to=1.0,
                                        orient=tk.HORIZONTAL)
        self.conf_threshold.set(0.25)  # Default confidence for SAHI
        self.conf_threshold.grid(row=0, column=3, sticky="ew", padx=5)

        # SAHI Toggle
        self.sahi_toggle = ttk.Checkbutton(
            controls_frame,
            text="Enable SAHI",
            variable=self.sahi_enabled,
            command=self.toggle_sahi
        )
        self.sahi_toggle.grid(row=0, column=4, padx=5)
        
        # Video controls
        self.play_button = ttk.Button(controls_frame,
                                      text="▶",
                                      style='Secondary.TButton',
                                      command=self.toggle_video,
                                      width=3)
        self.play_button.grid(row=0, column=5, padx=5)
        self.play_button.state(['disabled'])
        
        # Second row for SAHI controls
        ttk.Label(controls_frame,
                  text="Slice Size:",
                  style='Status.TLabel').grid(row=1, column=0, padx=5, pady=(10, 0), sticky="e")
                  
        slice_sizes = ["256", "512", "1024", "2048", "Auto"]
        self.slice_size_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.slice_size_var,
            values=slice_sizes,
            state="readonly",
            width=8
        )
        self.slice_size_combo.grid(row=1, column=1, padx=5, pady=(10, 0), sticky="w")
        self.slice_size_combo.current(2)  # Set default to 1024
        
        ttk.Label(controls_frame,
                  text="Overlap:",
                  style='Status.TLabel').grid(row=1, column=2, padx=5, pady=(10, 0), sticky="e")
                  
        overlap_scale = ttk.Scale(
            controls_frame,
            from_=0.1,
            to=0.5,
            orient=tk.HORIZONTAL,
            variable=self.slice_overlap_var,
            length=100
        )
        overlap_scale.grid(row=1, column=3, padx=5, pady=(10, 0), sticky="w")
        
        # Label to show overlap value
        self.overlap_label = ttk.Label(
            controls_frame,
            text="0.20",
            style='Status.TLabel'
        )
        self.overlap_label.grid(row=1, column=4, pady=(10, 0), sticky="w")
        
        # Update overlap label when scale changes
        overlap_scale.configure(command=self.update_overlap_label)

    def update_overlap_label(self, value=None):
        self.overlap_label.configure(text=f"{self.slice_overlap_var.get():.2f}")

    def toggle_sahi(self):
        # Check if SAHI is enabled and update UI accordingly
        if self.sahi_enabled.get():
            self.slice_size_combo.state(['!disabled'])
            self.status_var.set("SAHI enabled - Detection will use image slicing")
        else:
            self.slice_size_combo.state(['!disabled'])
            self.status_var.set("SAHI disabled - Using standard YOLO detection")

    def setup_canvas(self):
        # Canvas frame with white background
        canvas_frame = ttk.Frame(self.main_frame, style='Controls.TFrame')
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

        # Canvas with gray background to show bounds
        self.canvas = tk.Canvas(canvas_frame,
                                bg='#e0e0e0',
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    def setup_status_bar(self):
        status_frame = ttk.Frame(self.main_frame, style='Controls.TFrame')
        status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame,
                                 textvariable=self.status_var,
                                 style='Status.TLabel')
        status_label.grid(row=0, column=0, sticky="w")

        # Inference time display
        self.inference_var = tk.StringVar(value="Inference Time: --")
        inference_label = ttk.Label(status_frame,
                                   textvariable=self.inference_var,
                                   style='Status.TLabel')
        inference_label.grid(row=0, column=1, sticky="e")

    def load_model(self):
        try:
            self.status_var.set("Loading YOLO model...")
            self.root.update()

            # Find model in models directory
            model_path = os.path.join("models", "3k_TIP_s.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join("models", "best.pt")
                if not os.path.exists(model_path):
                    self.status_var.set("Error: Could not find model file. Please place the model in the models directory.")
                    return

            # Load YOLO model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device = "cpu"
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to(device)
            
            # Initialize SAHI detection model (will be used when SAHI is enabled)
            self.sahi_detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=model_path,
                confidence_threshold=self.conf_threshold.get(),
                device=device
            )

            self.status_var.set(f"Model loaded successfully (using {device.upper()})")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            # Stop any running video/webcam
            self.stop_video()

            self.current_file = file_path
            self.is_video = False
            self.is_webcam = False
            self.play_button.state(['disabled'])
            self.process_image(file_path)

    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            # Stop any running video/webcam
            self.stop_video()

            self.current_file = file_path
            self.is_video = True
            self.is_webcam = False
            self.video_source = cv2.VideoCapture(file_path)
            self.play_button.state(['!disabled'])
            self.is_video_playing = False
            self.play_button.configure(text="▶")

            # Show first frame
            ret, frame = self.video_source.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.process_image(frame_rgb, is_array=True)

    def stop_video(self):
        # Stop any running video/webcam
        self.is_video_playing = False
        self.play_button.configure(text="▶")

        # Release video source if it exists
        if self.video_source is not None:
            if hasattr(self.video_source, 'release'):
                self.video_source.release()
            self.video_source = None

        # Wait for thread to terminate
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)

    def process_image(self, image_source, is_array=False):
        try:
            self.status_var.set("Processing...")
            self.root.update()

            # Load and process image
            if is_array:
                image = Image.fromarray(image_source)
            else:
                image = Image.open(image_source)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Run inference
            processed_image = self.run_inference_on_image(image)

            # Display result
            self.display_image(processed_image)

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

    def determine_optimal_slice_size(self, image):
        """Determine the optimal slice size based on image resolution"""
        width, height = image.size
        
        # Calculate the average dimension
        avg_dimension = (width + height) / 2
        
        # Select slice size based on average dimension
        if avg_dimension < 1000:
            return 256
        elif avg_dimension < 2000:
            return 512
        elif avg_dimension < 4000:
            return 1024
        else:
            return 2048

    def run_inference_on_image(self, image):
        """Run inference on image with or without SAHI based on settings"""
        image_np = np.array(image)
        
        # Measure overall inference time
        total_start_time = time.time()
        
        if self.sahi_enabled.get():
            # Use SAHI for sliced inference
            try:
                # Update SAHI model confidence threshold
                self.sahi_detection_model.confidence_threshold = self.conf_threshold.get()
                
                # Determine slice size
                slice_size_text = self.slice_size_var.get()
                if slice_size_text == "Auto":
                    slice_size = self.determine_optimal_slice_size(image)
                else:
                    slice_size = int(slice_size_text)
                
                overlap_ratio = self.slice_overlap_var.get()
                
                # Get SAHI prediction
                sahi_start_time = time.time()
                result = get_sliced_prediction(
                    image_np,
                    self.sahi_detection_model,
                    slice_height=slice_size,
                    slice_width=slice_size,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio,
                )
                sahi_time = (time.time() - sahi_start_time) * 1000
                
                # Convert SAHI result to image with annotations
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Draw predictions
                for object_prediction in result.object_prediction_list:
                    # Get values
                    bbox = object_prediction.bbox
                    category_name = object_prediction.category.name
                    score = object_prediction.score.value
                    
                    # Draw bounding box
                    cv2.rectangle(
                        image_np, 
                        (int(bbox.minx), int(bbox.miny)), 
                        (int(bbox.maxx), int(bbox.maxy)), 
                        (0, 255, 0), 
                        4
                    )
                    
                    # Draw label with proper visibility
                    label_text = f"{category_name}: {score:.2f}"
                    font_scale = 1.0
                    thickness = 3
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # Get text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, font, font_scale, thickness
                    )
                    
                    # Draw background rectangle
                    padding = 8
                    cv2.rectangle(
                        image_np,
                        (int(bbox.minx), int(bbox.miny - text_height - padding * 2)),
                        (int(bbox.minx + text_width + padding * 2), int(bbox.miny)),
                        (0, 255, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        image_np,
                        label_text,
                        (int(bbox.minx) + padding, int(bbox.miny) - padding),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness
                    )
                
                # Calculate total time
                total_time = (time.time() - total_start_time) * 1000
                
                # Update inference time display
                self.inference_var.set(
                    f"SAHI: {sahi_time:.2f}ms, Total: {total_time:.2f}ms, Slices: {slice_size}px"
                )
                
            except Exception as e:
                self.status_var.set(f"SAHI error: {str(e)}")
                # Fall back to regular YOLO
                return self.run_yolo_inference(image_np, total_start_time)
                
        else:
            # Standard YOLO inference
            return self.run_yolo_inference(image_np, total_start_time)
            
        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    def run_yolo_inference(self, image_np, start_time=None):
        """Run standard YOLO inference without SAHI"""
        if start_time is None:
            start_time = time.time()
            
        yolo_start_time = time.time()
        results = self.yolo_model(image_np, conf=self.conf_threshold.get())
        yolo_time = (time.time() - yolo_start_time) * 1000
        
        # Update status but keep it brief during webcam/video playback
        if not self.is_video_playing:
            self.status_var.set(f"Standard YOLO detection completed")
            
        # Process results  
        for result in results:
            for box in result.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls)
                conf = float(box.conf)

                # Draw the bounding box
                cv2.rectangle(image_np, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 4)
                
                # Draw the label with better visibility
                label_text = f"{result.names[cls]}: {conf:.2f}"
                font_scale = 1.0
                thickness = 3
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Get text size for properly sized background
                (label_w, label_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # Draw background rectangle with padding
                padding = 8
                cv2.rectangle(
                    image_np, 
                    (b[0], b[1] - label_h - padding * 2),
                    (b[0] + label_w + padding * 2, b[1]),
                    (0, 255, 0), 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    image_np,
                    label_text,
                    (b[0] + padding, b[1] - padding),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness
                )
        
        # Calculate and update total inference time
        total_time = (time.time() - start_time) * 1000
        self.inference_var.set(f"YOLO: {yolo_time:.2f}ms, Total: {total_time:.2f}ms")
        
        return image_np

    def display_image(self, image_np):
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:  # Canvas not yet realized
            canvas_width = 800
            canvas_height = 600

        # Resize image to fit canvas while maintaining aspect ratio
        height, width = image_np.shape[:2]
        canvas_ratio = canvas_width / canvas_height
        image_ratio = width / height

        if image_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / image_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * image_ratio)

        image_np = cv2.resize(image_np, (new_width, new_height))

        # Convert to PhotoImage and display
        image = Image.fromarray(image_np)
        photo = ImageTk.PhotoImage(image=image)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                 image=photo,
                                 anchor=tk.CENTER)
        self.canvas.image = photo  # Keep a reference!

    def toggle_video(self):
        if self.is_video_playing:
            self.is_video_playing = False
            self.play_button.configure(text="▶")
        else:
            self.is_video_playing = True
            self.play_button.configure(text="⏸")
            if not self.video_thread or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self.process_video)
                self.video_thread.daemon = True
                self.video_thread.start()

    def process_video(self):
        while self.is_video_playing and self.video_source is not None:
            ret, frame = self.video_source.read()
            if not ret:
                self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.process_image(frame_rgb, is_array=True)

            # Control frame rate
            time.sleep(0.03)  # Approx. 30 FPS for video files


def main():
    root = tk.Tk()
    app = ModernSAHIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
