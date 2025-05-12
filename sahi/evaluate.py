import os
import json
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys
from tabulate import tabulate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def run_command(cmd, description):
    """Run a shell command and display output"""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    try:
        # Use explicit encoding and handle errors for Windows compatibility
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=False,  # Don't use universal_newlines for better encoding control
            bufsize=1
        )
        
        # Print output in real-time with encoding handling
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                try:
                    # Try utf-8 first
                    line = output.decode('utf-8').strip()
                except UnicodeDecodeError:
                    try:
                        # Fall back to cp1252 (Windows default)
                        line = output.decode('cp1252').strip()
                    except UnicodeDecodeError:
                        # Last resort: ignore errors
                        line = output.decode('utf-8', errors='ignore').strip()
                
                print(line)
        
        rc = process.poll()
        if rc == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with return code {rc}")
            return False
    except Exception as e:
        print(f"❌ Error executing {description}: {str(e)}")
        return False

def generate_coco_dataset():
    """Generate COCO format dataset from YOLO detections"""
    print("\n📊 Generating COCO dataset from YOLO predictions...")
    
    # Load YOLO model
    model_path = "models/3k_TIP_s.pt"
    if not os.path.exists(model_path):
        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            print("❌ Error: Model not found in models directory.")
            return False
    
    model = YOLO(model_path)
    image_dir = "image/"
    
    # Prepare COCO format
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 0,
                "name": "Litter",
                "supercategory": "waste"
            }
        ]
    }
    
    annotation_id = 1
    
    # Process all images in directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        print("❌ Error: No images found in image/ directory.")
        return False
    
    print(f"🔍 Processing {len(image_files)} images...")
    
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        
        # Run inference
        results = model(image_path)
        
        # Add image to COCO format
        image_data = {
            "id": idx + 1,
            "file_name": filename,
            "width": results[0].orig_shape[1],
            "height": results[0].orig_shape[0],
            "date_captured": time.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        }
        coco_format["images"].append(image_data)
        
        # Process detections
        for detection in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = detection
            
            # All detections are "Litter" class (id=0)
            class_id = 0
            
            # Convert to COCO bbox format [x, y, width, height]
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            annotation_data = {
                "id": annotation_id,
                "image_id": idx + 1,
                "category_id": class_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "segmentation": []
            }
            coco_format["annotations"].append(annotation_data)
            annotation_id += 1
    
    # Save dataset.json
    with open("dataset.json", "w") as json_file:
        json.dump(coco_format, json_file, indent=4)
    
    print(f"✅ COCO dataset generated with {len(coco_format['images'])} images and {len(coco_format['annotations'])} annotations")
    return True

def create_config_yaml():
    """Create or update config.yaml for SAHI"""
    config_content = """
model_type: yolov8
model_path: models/3k_TIP_s.pt
model_device: cuda:0 # or 'cpu'
model_confidence_threshold: 0.25
"""
    
    with open("config.yaml", "w") as f:
        f.write(config_content.strip())
    
    print("✅ Updated config.yaml for SAHI")
    return True

def calculate_coco_metrics(ground_truth_path, prediction_path):
    """Calculate COCO metrics using pycocotools"""
    try:
        # Initialize COCO ground truth
        coco_gt = COCO(ground_truth_path)
        
        # Load predictions
        coco_dt = coco_gt.loadRes(prediction_path)
        
        # Initialize COCO evaluator
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        
        # Evaluate
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract key metrics
        metrics = {
            'AP': coco_eval.stats[0],  # AP at IoU=0.50:0.95
            'AP50': coco_eval.stats[1],  # AP at IoU=0.50
            'AP75': coco_eval.stats[2],  # AP at IoU=0.75
            'APs': coco_eval.stats[3],   # AP for small objects
            'APm': coco_eval.stats[4],   # AP for medium objects
            'APl': coco_eval.stats[5],   # AP for large objects
        }
        
        return metrics
    except Exception as e:
        print(f"❌ Error calculating COCO metrics: {str(e)}")
        return None

def get_optimal_slice_size(image_dir):
    """Determine optimal slice size based on image resolutions in the directory"""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        return 512  # Default if no images found
    
    total_width = 0
    total_height = 0
    
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            total_width += width
            total_height += height
        except Exception:
            continue
    
    if len(image_files) > 0:
        avg_width = total_width / len(image_files)
        avg_height = total_height / len(image_files)
        avg_dimension = (avg_width + avg_height) / 2
        
        # Select slice size based on average dimension
        if avg_dimension < 800:
            return 256
        elif avg_dimension < 1600:
            return 512
        elif avg_dimension < 2400:
            return 1024
        else:
            return 2048
    
    return 512  # Default fallback

def run_evaluations(slice_size=None):
    """Run YOLO and SAHI evaluations"""
    # Step 1: Generate dataset.json
    if not generate_coco_dataset():
        return False
    
    # Step 2: Ensure config.yaml is correct
    if not create_config_yaml():
        return False
    
    # Determine slice size if not provided
    if slice_size is None or slice_size == "auto":
        slice_size = get_optimal_slice_size("image/")
        print(f"🔍 Automatically selected slice size: {slice_size}px based on image dimensions")
    else:
        slice_size = int(slice_size)
        print(f"🔍 Using requested slice size: {slice_size}px")
    
    # Step 3: Run standard YOLO prediction
    model_path = "models/3k_TIP_s.pt"
    if not os.path.exists(model_path):
        model_path = "models/best.pt"

    # Run YOLO detection and create our own result.json instead of relying on YOLO's output
    yolo_cmd = f"yolo detect predict model={model_path} source=image/ imgsz=640 conf=0.25 save=True"
    if not run_command(yolo_cmd, "YOLO prediction"):
        return False
    
    # Manually create a result.json file for YOLO results in COCO format
    print("Creating YOLO result.json file in COCO format...")
    
    # Find the latest YOLO prediction directory
    yolo_pred_dirs = list(Path("runs/detect").glob("predict*"))
    if not yolo_pred_dirs:
        print("❌ Error: No YOLO prediction directory found.")
        return False
    
    latest_yolo_dir = str(sorted(yolo_pred_dirs, key=os.path.getmtime)[-1])
    print(f"Found YOLO prediction directory: {latest_yolo_dir}")
    
    # Load YOLO model and generate result.json manually
    model = YOLO(model_path)
    yolo_results = []
    
    # Process all images in the image directory
    image_dir = "image/"
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.join(latest_yolo_dir, "labels"), exist_ok=True)
    
    # Load dataset.json to get image dimensions
    with open("dataset.json", "r") as f:
        dataset = json.load(f)
    
    # Create image_id to dimensions mapping
    image_info = {}
    for img in dataset["images"]:
        image_info[img["file_name"]] = {
            "id": img["id"],
            "width": img["width"],
            "height": img["height"]
        }
    
    # Process each image and generate COCO format results
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        results = model(image_path, conf=0.25)
        
        img_data = image_info.get(filename, {"id": 1, "width": 640, "height": 640})
        
        for result in results:
            for box in result.boxes:
                b = box.xyxy[0].cpu().numpy().astype(float)
                conf = float(box.conf)
                
                # Create detection in COCO format
                detection = {
                    "image_id": img_data["id"],
                    "category_id": 0,  # Litter class
                    "bbox": [
                        float(b[0]),  # x
                        float(b[1]),  # y
                        float(b[2] - b[0]),  # width
                        float(b[3] - b[1])   # height
                    ],
                    "score": conf
                }
                yolo_results.append(detection)
    
    # Save YOLO results in COCO format
    yolo_result_path = os.path.join(latest_yolo_dir, "result.json")
    with open(yolo_result_path, "w") as f:
        json.dump(yolo_results, f, indent=4)
    
    print(f"✅ Created YOLO result.json with {len(yolo_results)} detections")
    
    # Step 4: Run SAHI prediction WITH slicing
    model_path_escaped = model_path.replace("\\", "/")  # Ensure forward slashes for SAHI
    sahi_predict_sliced_cmd = f'sahi predict --source "image/" --dataset_json_path "dataset.json" --model_type yolov8 --model_path "{model_path_escaped}" --model_config_path "config.yaml" --slice_height {slice_size} --slice_width {slice_size} --overlap_height_ratio 0.2 --overlap_width_ratio 0.2'
    if not run_command(sahi_predict_sliced_cmd, f"SAHI prediction (with {slice_size}x{slice_size} slicing)"):
        return False
    
    # Find the latest SAHI sliced result
    sahi_sliced_results = list(Path("runs/predict").glob("exp*/result.json"))
    if not sahi_sliced_results:
        print("❌ Error: No SAHI result.json found.")
        return False
    
    sahi_sliced_result_path = str(sorted(sahi_sliced_results, key=os.path.getmtime)[-1])
    sahi_sliced_result_path_escaped = sahi_sliced_result_path.replace("\\", "/")
    print(f"📄 Found SAHI sliced result.json: {sahi_sliced_result_path}")
    
    # Copy all result files to a central location for easier access
    os.makedirs("results", exist_ok=True)
    import shutil
    shutil.copy(yolo_result_path, "results/yolo_result.json")
    shutil.copy(sahi_sliced_result_path, f"results/sahi_sliced_{slice_size}_result.json")
    
    # Step 5: Run SAHI visualization commands
    # These don't depend on metrics but just visualize the detections
    yolo_result_path_escaped = yolo_result_path.replace("\\", "/")
    yolo_analyze_cmd = f'sahi coco analyse --dataset_json_path "dataset.json" --result_json_path "{yolo_result_path_escaped}"'
    run_command(yolo_analyze_cmd, "YOLO analysis")
    
    sahi_sliced_analyze_cmd = f'sahi coco analyse --dataset_json_path "dataset.json" --result_json_path "{sahi_sliced_result_path_escaped}"'
    run_command(sahi_sliced_analyze_cmd, f"SAHI analysis (with {slice_size}x{slice_size} slicing)")
    
    # Create improved comparison plots
    create_improved_plots(yolo_result_path, sahi_sliced_result_path, slice_size)
    
    return True

def create_improved_plots(yolo_result_path, sahi_result_path, slice_size):
    """Create improved comparison plots between YOLO and SAHI results"""
    print("\n📊 Generating improved comparison charts...")
    
    try:
        # Load results
        with open(yolo_result_path, 'r') as f:
            yolo_results = json.load(f)
            
        with open(sahi_result_path, 'r') as f:
            sahi_results = json.load(f)
        
        # Count detections
        yolo_count = len(yolo_results)
        sahi_count = len(sahi_results)
        
        # Create charts directory if it doesn't exist
        os.makedirs("charts", exist_ok=True)
        
        # 1. Create improved detection count comparison
        plt.figure(figsize=(10, 6))
        methods = ['YOLO', f'SAHI (Sliced {slice_size}x{slice_size})']
        counts = [yolo_count, sahi_count]
        
        # Calculate improvement percentage
        improvement_pct = ((sahi_count - yolo_count) / yolo_count * 100) if yolo_count > 0 else 0
        
        # Set y-axis limit with padding to ensure labels stay within bounds
        y_max = max(counts) * 1.2  # 20% padding
        
        # Create bar chart
        ax = plt.gca()
        colors = ['#3498db', '#e74c3c']
        bars = plt.bar(methods, counts, color=colors)
        plt.title('Detection Count Comparison', fontsize=16)
        plt.ylabel('Number of Detections', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, y_max)  # Set y-axis limit with padding
        
        # Add count labels on top of bars
        for i, (count, bar) in enumerate(zip(counts, bars)):
            plt.text(bar.get_x() + bar.get_width()/2., count + (y_max * 0.02),
                     f'{count}', ha='center', fontsize=12, fontweight='bold')
        
        # Add improvement percentage
        if yolo_count > 0:
            # Position the improvement text at 40% height of the second bar
            plt.text(1, counts[1] * 0.6, f'+{improvement_pct:.1f}%', 
                     fontsize=14, color='darkgreen', fontweight='bold')
            
            # Add arrow for improvement
            plt.annotate('', xy=(1, counts[1] * 0.8), xytext=(0, counts[0] * 0.8),
                        arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
        
        plt.tight_layout()
        plt.savefig('charts/detection_counts.png')
        
        # 2. Create confidence score distribution
        plt.figure(figsize=(15, 5))
        
        # Get confidence scores
        yolo_scores = [det.get('score', 0) for det in yolo_results]
        sahi_scores = [det.get('score', 0) for det in sahi_results]
        
        plt.subplot(1, 2, 1)
        plt.hist(yolo_scores, bins=20, alpha=0.7, color='#3498db')
        plt.title('YOLO Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(sahi_scores, bins=20, alpha=0.7, color='#e74c3c')
        plt.title(f'SAHI (Sliced {slice_size}x{slice_size}) Confidence Distribution')
        plt.xlabel('Confidence Score')
        
        plt.tight_layout()
        plt.savefig('charts/confidence_distribution.png')
        
        # 3. Create box size distribution
        plt.figure(figsize=(12, 5))
        
        # Calculate bbox areas for each method
        def get_areas(results):
            areas = []
            for det in results:
                if 'bbox' in det:
                    bbox = det['bbox']
                    if len(bbox) >= 4:
                        # COCO format: [x, y, width, height]
                        area = bbox[2] * bbox[3]
                        areas.append(area)
            return areas
        
        yolo_areas = get_areas(yolo_results)
        sahi_areas = get_areas(sahi_results)
        
        plt.subplot(1, 2, 1)
        plt.hist(yolo_areas, bins=20, alpha=0.7, color='#3498db')
        plt.title('YOLO Box Size Distribution')
        plt.xlabel('Bounding Box Area (pixels²)')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.hist(sahi_areas, bins=20, alpha=0.7, color='#e74c3c')
        plt.title(f'SAHI (Sliced {slice_size}x{slice_size}) Box Size Distribution')
        plt.xlabel('Bounding Box Area (pixels²)')
        
        plt.tight_layout()
        plt.savefig('charts/box_size_distribution.png')
        
        # 4. Create a simple detection summary table
        plt.figure(figsize=(8, 4))
        table_data = [
            ["Method", "Detections", "Improvement"],
            ["YOLO", str(yolo_count), ""],
            [f"SAHI (Sliced {slice_size}x{slice_size})", 
             str(sahi_count), 
             f"+{improvement_pct:.1f}%"]
        ]
        
        # Create a table
        ax = plt.gca()
        ax.axis('tight')
        ax.axis('off')
        
        table = plt.table(
            cellText=table_data[1:],  # Skip the header row for cellText
            colLabels=table_data[0],  # Use the first row as column labels
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Color the improvement cell
        improvement_cell = table[2, 2]  # Row 2, Col 2 (0-indexed after removing header)
        improvement_cell.set_facecolor('#d5f5e3')  # Light green
        
        # Add a title
        plt.title(f"Detection Summary (Slice Size: {slice_size}px)", fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig('charts/detection_summary.png', bbox_inches='tight')
        
        print(f"✅ Improved comparison charts saved to charts/ directory:")
        print("   - detection_counts.png")
        print("   - confidence_distribution.png")
        print("   - box_size_distribution.png")
        print("   - detection_summary.png")
        
    except Exception as e:
        print(f"❌ Error creating improved comparison plots: {str(e)}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SAHI YOLO Litter Detection Evaluation")
    parser.add_argument("--slice_size", default="auto", 
                      help="Slice size for SAHI (256, 512, 1024, 2048, or 'auto')")
    
    # Check if script is run directly (not imported)
    if __name__ == "__main__":
        args = parser.parse_args()
    else:
        # Create a default object for module import case
        args = type('Args', (), {"slice_size": "auto"})
    
    print("=" * 80)
    print("🔍 Litter Detection Comparison: YOLO vs SAHI")
    print("=" * 80)
    
    # Convert slice_size to the right type
    slice_size = args.slice_size
    if slice_size != "auto":
        try:
            slice_size = int(slice_size)
            # Validate slice size is in expected range
            valid_sizes = [256, 512, 1024, 2048]
            if slice_size not in valid_sizes:
                closest = min(valid_sizes, key=lambda x: abs(x - slice_size))
                print(f"⚠️ Warning: {slice_size} is not a standard slice size. Using closest size: {closest}")
                slice_size = closest
        except ValueError:
            print(f"⚠️ Warning: Invalid slice size '{slice_size}'. Using 'auto' instead.")
            slice_size = "auto"
    
    run_evaluations(slice_size)
    
    print("\n" + "=" * 80)
    print("✅ Evaluation complete! Check the results in:")
    print("   - runs/detect/: YOLO detection results")
    print("   - runs/predict/: SAHI prediction results")
    print("   - charts/: Comparison visualizations")
    print("=" * 80)

if __name__ == "__main__":
    main()