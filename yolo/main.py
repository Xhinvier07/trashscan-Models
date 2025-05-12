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


class ModernYOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Litter Detection")
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
        self.tracking_enabled = False
        self.tracker = None
        self.track_history = {}  # Store tracking history
        self.webcam_id = 0  # Default webcam ID

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
        controls_frame.grid_columnconfigure(3, weight=1)

        # Input controls
        ttk.Button(controls_frame,
                   text="Open Image",
                   style='Primary.TButton',
                   command=self.load_image).grid(row=0, column=0, padx=5)

        ttk.Button(controls_frame,
                   text="Open Video",
                   style='Primary.TButton',
                   command=self.load_video).grid(row=0, column=1, padx=5)

        # Webcam button
        ttk.Button(controls_frame,
                   text="Use Webcam",
                   style='Primary.TButton',
                   command=self.start_webcam).grid(row=0, column=2, padx=5)

        # Tracking toggle button
        self.tracking_button = ttk.Button(
            controls_frame,
            text="Tracking: OFF",
            style='Toggle.TButton',
            command=self.toggle_tracking
        )
        self.tracking_button.grid(row=0, column=3, padx=5)

        # Confidence threshold
        ttk.Label(controls_frame,
                  text="Confidence:",
                  style='Status.TLabel').grid(row=0, column=4, padx=(20, 5))

        self.conf_threshold = ttk.Scale(controls_frame,
                                        from_=0.0,
                                        to=1.0,
                                        orient=tk.HORIZONTAL)
        self.conf_threshold.set(0.5)
        self.conf_threshold.grid(row=0, column=5, sticky="ew", padx=5)

        # Video controls
        self.play_button = ttk.Button(controls_frame,
                                      text="▶",
                                      style='Secondary.TButton',
                                      command=self.toggle_video,
                                      width=3)
        self.play_button.grid(row=0, column=6, padx=5)
        self.play_button.state(['disabled'])

        # Create a second row for webcam controls
        controls_frame.grid_rowconfigure(1, weight=0)

        # Webcam selector
        ttk.Label(controls_frame,
                  text="Webcam ID:",
                  style='Status.TLabel').grid(row=1, column=0, padx=5, pady=(10, 0), sticky="e")

        self.webcam_id_var = tk.StringVar(value="0")
        webcam_combobox = ttk.Combobox(controls_frame,
                                       textvariable=self.webcam_id_var,
                                       values=["0", "1", "2", "3"],
                                       width=5,
                                       state="readonly")
        webcam_combobox.grid(row=1, column=1, padx=5, pady=(10, 0), sticky="w")
        webcam_combobox.bind("<<ComboboxSelected>>", self.on_webcam_change)

    def on_webcam_change(self, event):
        # Update webcam ID when combobox selection changes
        self.webcam_id = int(self.webcam_id_var.get())
        if self.is_webcam and self.is_video_playing:
            # Restart webcam with new ID if currently using webcam
            self.stop_video()
            time.sleep(0.5)  # Give time for the previous webcam to close
            self.start_webcam()

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

        # FPS counter
        self.fps_var = tk.StringVar(value="FPS: --")
        fps_label = ttk.Label(status_frame,
                              textvariable=self.fps_var,
                              style='Status.TLabel')
        fps_label.grid(row=0, column=1, sticky="e")

    def load_model(self):
        try:
            self.status_var.set("Loading model...")
            self.root.update()

            # Load YOLO model
            self.model = YOLO('models/3k_TIP_s.pt')  # Use your model path here

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)

            self.status_var.set(f"Model loaded successfully (using {device.upper()})")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")

    def toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled
        if self.tracking_enabled:
            self.tracking_button.configure(text="Tracking: ON")
            # Reset track history when enabling tracking
            self.track_history = {}
            self.status_var.set("Object tracking enabled")
        else:
            self.tracking_button.configure(text="Tracking: OFF")
            self.status_var.set("Object tracking disabled")

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
            # Reset track history
            self.track_history = {}
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
            # Reset track history
            self.track_history = {}

            # Show first frame
            ret, frame = self.video_source.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.process_image(frame_rgb, is_array=True)

    def start_webcam(self):
        # Stop any running video/webcam
        self.stop_video()

        try:
            # Try to open the webcam
            webcam_id = int(self.webcam_id_var.get())
            self.video_source = cv2.VideoCapture(webcam_id)

            if not self.video_source.isOpened():
                self.status_var.set(f"Error: Cannot open webcam with ID {webcam_id}")
                return

            self.is_webcam = True
            self.is_video = True
            self.current_file = f"Webcam {webcam_id}"
            self.play_button.state(['!disabled'])
            self.is_video_playing = True
            self.play_button.configure(text="⏸")
            # Reset track history
            self.track_history = {}

            self.status_var.set(f"Webcam {webcam_id} activated")

            # Start webcam processing thread
            if not self.video_thread or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self.process_video)
                self.video_thread.daemon = True
                self.video_thread.start()

        except Exception as e:
            self.status_var.set(f"Error starting webcam: {str(e)}")

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

            if not self.is_video:
                self.status_var.set("Ready")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

    def run_inference_on_image(self, image):
        image_np = np.array(image)
        start_time = time.time()

        if self.tracking_enabled and self.is_video:
            # Use tracking mode
            results = self.model.track(
                image_np,
                conf=self.conf_threshold.get(),
                persist=True,
                tracker="bytetrack.yaml"
            )
            tracking_active = " with tracking" if self.tracking_enabled else ""
            inference_time = (time.time() - start_time) * 1000

            # Update status but keep it brief during webcam/video playback
            if not self.is_video_playing:
                self.status_var.set(f"Inference Time{tracking_active}: {inference_time:.2f} ms")

            # Process tracking results
            for result in results:
                if result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                    clss = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy().astype(float)

                    # Draw each tracked box with its trail
                    for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                        # Add current position to track history
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append(
                            (box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2))

                        # Limit trail length
                        if len(self.track_history[track_id]) > 30:
                            self.track_history[track_id] = self.track_history[track_id][-30:]

                        # Draw object box
                        cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)

                        # Draw ID and class label
                        label_text = f"ID:{track_id} {result.names[cls]}: {conf:.2f}"
                        font_scale = 1.0
                        thickness = 2
                        (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        cv2.rectangle(image_np, (box[0], box[1] - label_h - 16), (box[0] + label_w + 16, box[1]), (0, 255, 0), -1)
                        cv2.putText(image_np, label_text, (box[0] + 8, box[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

                        # Draw trail
                        for i in range(1, len(self.track_history[track_id])):
                            trail_color = (255, 0, 0)  # BGR format → (Blue=0, Green=0, Red=255) → PURE RED
                            thickness = 2
                            cv2.line(image_np, self.track_history[track_id][i - 1], self.track_history[track_id][i],
                                     trail_color, thickness)

        else:
            # Standard detection mode
            results = self.model(image_np, conf=self.conf_threshold.get())
            inference_time = (time.time() - start_time) * 1000

            # Update status but keep it brief during webcam/video playback
            if not self.is_video_playing:
                self.status_var.set(f"Inference Time: {inference_time:.2f} ms")

            # Process standard detection results
            for result in results:
                for box in result.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls)
                    conf = float(box.conf)

                    cv2.rectangle(image_np, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 4)
                    label_text = f"{result.names[cls]}: {conf:.2f}"
                    font_scale = 1.0
                    thickness = 2
                    (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(image_np, (b[0], b[1] - label_h - 16), (b[0] + label_w + 16, b[1]), (0, 255, 0), -1)
                    cv2.putText(image_np, label_text, (b[0] + 8, b[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Calculate and update FPS for video/webcam
        if self.is_video_playing:
            fps = 1.0 / (time.time() - start_time)
            self.fps_var.set(f"FPS: {fps:.1f}")

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
            # Reset track history when starting video
            if self.tracking_enabled:
                self.track_history = {}

            if not self.video_thread or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self.process_video)
                self.video_thread.daemon = True
                self.video_thread.start()

    def process_video(self):
        last_time = time.time()
        frame_count = 0

        while self.is_video_playing and self.video_source is not None:
            ret, frame = self.video_source.read()
            if not ret:
                if self.is_webcam:
                    # For webcam, if we can't read a frame, try to reconnect
                    self.status_var.set(f"Webcam connection lost. Trying to reconnect...")
                    self.video_source.release()
                    time.sleep(1.0)
                    self.video_source = cv2.VideoCapture(int(self.webcam_id_var.get()))
                    continue
                else:
                    # For video file, loop back to the beginning
                    self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Reset track history when video loops
                    if self.tracking_enabled:
                        self.track_history = {}
                    continue

            # Process the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # If it's a webcam and the image is flipped, fix it
            if self.is_webcam:
                frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror webcam image horizontally

            self.process_image(frame_rgb, is_array=True)

            # Update FPS calculation every second
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                self.fps_var.set(f"FPS: {fps:.1f}")
                last_time = current_time
                frame_count = 0

            # Control frame rate
            if not self.is_webcam:  # Don't add additional delay for webcam
                time.sleep(0.03)  # Approx. 30 FPS for video files


def main():
    root = tk.Tk()
    app = ModernYOLOApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()