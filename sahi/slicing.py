import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
import json

def calculate_slice_bboxes(image_height, image_width, slice_height, slice_width, 
                           overlap_height_ratio, overlap_width_ratio):
    """Calculate slice bounding boxes for an image"""
    slice_bboxes = []
    y_max = y_min = 0
    
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes

def main():
    st.set_page_config(page_title="Image Slicing Visualizer", layout="wide")
    
    st.title("Image Slicing Visualizer")
    st.write("Upload an image and visualize how it will be sliced for object detection")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Optional COCO JSON uploader for annotations
    coco_file = st.file_uploader("Optional: Upload COCO JSON for annotations", type=["json"])
    
    if coco_file is not None:
        try:
            coco_data = json.load(coco_file)
            st.success(f"Loaded COCO annotations with {len(coco_data.get('annotations', []))} annotations")
        except Exception as e:
            st.error(f"Error loading COCO file: {str(e)}")
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        width, height = image.size
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, caption=f"Original Size: {width}x{height}", use_column_width=True)
        
        # Slicing parameters
        st.subheader("Slicing Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            slice_width = st.slider("Slice Width", min_value=256, max_value=2048, value=512, step=128)
            overlap_width_ratio = st.slider("Width Overlap Ratio", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        
        with col2:
            slice_height = st.slider("Slice Height", min_value=256, max_value=2048, value=512, step=128)
            overlap_height_ratio = st.slider("Height Overlap Ratio", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
        
        # Calculate slice bboxes
        try:
            slices = calculate_slice_bboxes(
                image_height=height,
                image_width=width,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio
            )
            
            num_slices = len(slices)
            st.write(f"Number of slices: {num_slices}")
            
            # Preset color schemes
            color_schemes = {
                "Red Grid": {"grid": "#FF0000", "text_bg": "#FF0000", "text": "white"},
                "Blue Grid": {"grid": "#0000FF", "text_bg": "#0000FF", "text": "white"},
                "Green Grid": {"grid": "#00AA00", "text_bg": "#00AA00", "text": "white"},
                "High Contrast": {"grid": "#FFFF00", "text_bg": "#000000", "text": "white"}
            }
            
            selected_scheme = st.selectbox("Color Scheme", list(color_schemes.keys()))
            colors = color_schemes[selected_scheme]
            
            # Visualization options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_indices = st.checkbox("Show Slice Numbers", value=True)
            with col2:
                grid_thickness = st.slider("Grid Line Thickness", 1, 5, 2)
            with col3:
                image_quality = st.slider("Image Quality", 100, 300, 150, 50)
                
            # Visualize slicing grid
            st.subheader("Slicing Visualization")
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Display the image
            ax.imshow(np.array(image))
            
            # Add slicing grid
            for i, slice_bbox in enumerate(slices):
                x, y, width, height = slice_bbox[0], slice_bbox[1], slice_bbox[2] - slice_bbox[0], slice_bbox[3] - slice_bbox[1]
                rect = patches.Rectangle(
                    (x, y), width, height, 
                    linewidth=grid_thickness, edgecolor=colors["grid"], facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add slice number at the center of each slice
                if show_indices:
                    ax.text(
                        x + width/2, y + height/2, 
                        f"{i + 1}", 
                        color=colors["text"], fontsize=10, ha='center', va='center',
                        bbox=dict(facecolor=colors["text_bg"], alpha=0.7)
                    )
            
            ax.set_title(f"Slicing Grid: {num_slices} slices", fontsize=14)
            ax.axis('off')
            
            # Display the plot
            st.pyplot(fig)
            
            # Option to download the visualization
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=image_quality)
            buf.seek(0)
            
            st.download_button(
                label="Download Slicing Visualization",
                data=buf,
                file_name=f"slicing_visualization_{slice_width}x{slice_height}.png",
                mime="image/png"
            )
            
            # Information about slices
            with st.expander("Slice Information"):
                for i, slice_bbox in enumerate(slices):
                    x, y, width, height = slice_bbox[0], slice_bbox[1], slice_bbox[2] - slice_bbox[0], slice_bbox[3] - slice_bbox[1]
                    st.write(f"Slice {i+1}: Position [x={x}, y={y}], Size [{width}x{height}]")
            
        except Exception as e:
            st.error(f"Error calculating slices: {str(e)}")
            st.info("Please check your slicing parameters and try again.")

if __name__ == "__main__":
    main()
