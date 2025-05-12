import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import torch
from PIL import Image, ImageTk
import numpy as np
import threading
from queue import Queue
import time
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog


class ModernDetectron2App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detectron2 Litter Detection (faster_rcnn_R_50_FPN_3x)")
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

        style.configure('Main.TFrame', background='#f0f0f0')
        style.configure('Controls.TFrame', background='#ffffff', relief='flat')
        style.configure('Primary.TButton', padding=10, font=('Helvetica', 10))
        style.configure('Secondary.TButton', padding=8, font=('Helvetica', 10))
        style.configure('Status.TLabel', background='#ffffff', font=('Helvetica', 10))

    def setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, style='Main.TFrame', padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.setup_controls()
        self.setup_canvas()
        self.setup_status_bar()

    def setup_controls(self):
        controls_frame = ttk.Frame(self.main_frame, style='Controls.TFrame', padding="10")
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls_frame.grid_columnconfigure(2, weight=1)

        ttk.Button(controls_frame, text="Open Image", style='Primary.TButton',
                   command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Button(controls_frame, text="Open Video", style='Primary.TButton',
                   command=self.load_video).grid(row=0, column=1, padx=5)

        ttk.Label(controls_frame, text="Confidence:", style='Status.TLabel').grid(
            row=0, column=2, padx=(20, 5))

        self.conf_threshold = ttk.Scale(controls_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL)
        self.conf_threshold.set(0.5)
        self.conf_threshold.grid(row=0, column=3, sticky="ew", padx=5)

        self.play_button = ttk.Button(controls_frame, text="▶", style='Secondary.TButton',
                                     command=self.toggle_video, width=3)
        self.play_button.grid(row=0, column=4, padx=5)
        self.play_button.state(['disabled'])

    def setup_canvas(self):
        canvas_frame = ttk.Frame(self.main_frame, style='Controls.TFrame')
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg='#e0e0e0', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

    def setup_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.main_frame, textvariable=self.status_var,
                                style='Status.TLabel')
        status_label.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def register_custom_metadata(self):
        """Register metadata with proper class and color mapping"""
        DatasetCatalog.register("litter_dataset", lambda: [])
        MetadataCatalog.get("litter_dataset").set(
            thing_classes=["litter", "litter"],  # Keep two classes as specified
            thing_colors=[(0, 255, 0), (0, 255, 0)]  # Green color for both classes
        )

    def setup_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        #cfg.merge_from_file("detectron2/config.yml")
        cfg.MODEL.WEIGHTS = "detectron2/model_final.pth"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TRAIN = ("litter_dataset",)
        cfg.DATASETS.TEST = ()

        # Uncomment this to force CPU usage
        # cfg.MODEL.DEVICE = "cpu"

        # Uncomment this to enable CPU
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # cfg.MODEL.DEVICE = "cpu"

        return cfg

    def load_model(self):
        try:
            self.status_var.set("Loading model...")
            self.root.update()

            # Register custom metadata
            self.register_custom_metadata()

            # Setup configuration
            self.cfg = self.setup_cfg()
            device = self.cfg.MODEL.DEVICE

            # Create predictor
            self.predictor = DefaultPredictor(self.cfg)

            self.status_var.set(f"Model loaded successfully (using {device.upper()})")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")

    def process_single_image(self, frame):
        """Process image with proper color handling and timing"""
        # Measure inference time
        start_time = time.time()

        outputs = self.predictor(frame)

        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.status_var.set(f"Inference Time: {inference_time:.2f} ms")

        instances = outputs["instances"].to("cpu")

        # Filter instances based on confidence threshold
        instances = instances[instances.scores > self.conf_threshold.get()]

        # Use RGB format for visualization
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use custom visualization with enhanced visibility
        enhanced_image = self.draw_enhanced_predictions(rgb_frame, instances)

        # Convert back to BGR for OpenCV
        return cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)

    def draw_enhanced_predictions(self, image, instances):
        """Custom drawing function for more visible bounding boxes and labels"""
        # Get a copy of the image to draw on
        output = image.copy()
        
        # Get metadata
        metadata = MetadataCatalog.get("litter_dataset")
        
        # Draw boxes and labels
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
        scores = instances.scores.numpy() if instances.has("scores") else None
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
        
        if boxes is not None:
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                # Convert box to integers
                box = box.astype(np.int32)
                
                # Draw thicker bounding box (green)
                cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
                
                # Create label with class name and score
                class_name = metadata.thing_classes[cls] if metadata.thing_classes else f"Class {cls}"
                label_text = f"{class_name}: {score:.2f}"
                
                # Enhanced text parameters - larger font and thicker lines
                font_scale = 0.7
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                
                # Draw background rectangle for label with padding
                padding = 8
                cv2.rectangle(output, 
                             (box[0], box[1] - text_height - padding * 2), 
                             (box[0] + text_width + padding * 2, box[1]), 
                             (0, 255, 0), -1)  # Filled rectangle
                
                # Draw text (black on green background)
                cv2.putText(output, label_text, 
                           (box[0] + padding, box[1] - padding), 
                           font, font_scale, (0, 0, 0), thickness)
        
        return output

    def process_image(self, image_source, is_array=False):
        try:
            self.status_var.set("Processing...")
            self.root.update()

            # Load and process image
            if is_array:
                image = image_source
            else:
                image = cv2.imread(image_source)

            if image is None:
                raise Exception("Failed to load image")

            # Process image (inference timing is handled in process_single_image)
            processed_image = self.process_single_image(image)

            # Convert BGR to RGB for display
            display_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            # Display result
            self.display_image(display_image)

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")

    def display_image(self, image_np):
        # Image is already in RGB format here
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600

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
        image = Image.fromarray(image_np)  # Image is already in RGB
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                 image=photo, anchor=tk.CENTER)
        self.canvas.image = photo

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

            ret, frame = self.video_source.read()
            if ret:
                self.process_image(frame, is_array=True)

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

            self.process_image(frame, is_array=True)
            time.sleep(0.03)  # Approx. 30 FPS


def main():
    root = tk.Tk()
    app = ModernDetectron2App(root)
    root.mainloop()


if __name__ == "__main__":
    main()