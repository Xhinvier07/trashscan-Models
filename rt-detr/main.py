import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import numpy as np
import threading
from queue import Queue
import time
import os
from pathlib import Path


class ModernRTDETRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RT-DETR Litter Detection")
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
        controls_frame.grid_columnconfigure(2, weight=1)

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
        self.conf_threshold.set(0.5)
        self.conf_threshold.grid(row=0, column=3, sticky="ew", padx=5)

        # Video controls
        self.play_button = ttk.Button(controls_frame,
                                      text="▶",
                                      style='Secondary.TButton',
                                      command=self.toggle_video,
                                      width=3)
        self.play_button.grid(row=0, column=4, padx=5)
        self.play_button.state(['disabled'])

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
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.main_frame,
                                 textvariable=self.status_var,
                                 style='Status.TLabel')
        status_label.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def load_model(self):
        try:
            model_path = "models"
            self.status_var.set("Loading model...")
            self.root.update()



            # Uncomment this to enable CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device = "cpu"

            self.image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device
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
            self.current_file = file_path
            self.is_video = False
            self.play_button.state(['disabled'])
            self.process_image(file_path)

    def load_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if file_path:
            self.current_file = file_path
            self.is_video = True
            self.video_source = cv2.VideoCapture(file_path)
            self.play_button.state(['!disabled'])
            self.is_video_playing = False
            self.play_button.configure(text="▶")

            # Show first frame
            ret, frame = self.video_source.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.process_image(frame_rgb, is_array=True)

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

            # Don't overwrite the inference time message set by run_inference_on_image
            if not self.is_video and not self.status_var.get().startswith("Inference Time"):
                self.status_var.set("Ready")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")


    def run_inference_on_image(self, image):
        # Prepare input
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device=self.model.device, dtype=next(self.model.parameters()).dtype)
                  for k, v in inputs.items()}

        # Measure inference time
        start_time = time.time()  # Start time
        with torch.no_grad():
            outputs = self.model(**inputs)
        end_time = time.time()  # End time

        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        self.status_var.set(f"Inference Time: {inference_time:.2f} ms")  # Show in status bar

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]], device=self.model.device)
        results = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.conf_threshold.get(),
            target_sizes=target_sizes
        )[0]

        # Draw results
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        for box, score, label in zip(results["boxes"].cpu().numpy(),
                                     results["scores"].cpu().numpy(),
                                     results["labels"].cpu().numpy()):
            box = box.astype(int)
            # Increase box thickness from 2 to 4
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)

            label_text = f"Class {label}: {score:.2f}"
            
            # Increase font scale from 0.5 to 1.0 and thickness from 2 to 3
            font_scale = 0.7
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Get text size for properly sized background
            (label_w, label_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            
            # Add more padding (16 pixels) to the background rectangle
            padding = 8
            cv2.rectangle(image_np,
                          (box[0], box[1] - label_h - padding * 2),
                          (box[0] + label_w + padding * 2, box[1]),
                          (0, 255, 0),
                          -1)

            # Position text with better padding
            cv2.putText(image_np,
                        label_text,
                        (box[0] + padding, box[1] - padding),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness)

        return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

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
        while self.is_video_playing:
            ret, frame = self.video_source.read()
            if not ret:
                self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.process_image(frame_rgb, is_array=True)

            # Control frame rate
            time.sleep(0.30)  # Approx. 30 FPS


def main():
    root = tk.Tk()
    app = ModernRTDETRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()