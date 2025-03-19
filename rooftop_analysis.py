import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict
import cv2

# Load both models
rooftop_model = YOLO("model/rooftops.pt")
solar_panel_model = YOLO("model/solar_panel.pt")

# Initialize trackers for both objects
rooftop_tracker = sv.ByteTrack()
solar_tracker = sv.ByteTrack()

# Initialize annotators
box_annotator = sv.BoxAnnotator(thickness=3)
mask_annotator = sv.MaskAnnotator(opacity=0.3, color=sv.Color.GREEN)
label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

# Store rooftops with/without solar panels
rooftop_status = defaultdict(bool)  # Default is False (no solar panel)

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1, box2: Each box is [x1, y1, x2, y2] where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner
    
    Returns:
    float: IoU value
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Add these variables outside the callback function (global scope)
all_rooftops = set()  # To track all unique rooftop IDs
rooftops_with_solar = set()  # To track rooftop IDs that have solar panels

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    # Access the global tracking variables
    global all_rooftops, rooftops_with_solar

    # Get detections from both models
    rooftop_results = rooftop_model(frame, conf=0.86)[0]
    solar_results = solar_panel_model(frame)[0]
    
    # Convert to Supervision detections
    rooftop_detections = sv.Detections.from_ultralytics(rooftop_results)
    solar_detections = sv.Detections.from_ultralytics(solar_results)
    
    # Update trackers
    rooftop_detections = rooftop_tracker.update_with_detections(rooftop_detections)
    solar_detections = solar_tracker.update_with_detections(solar_detections)
    
    # Create a working copy of the frame
    annotated_frame = frame.copy()
    
    # Update our global tracking sets
    for r_idx, rooftop_id in enumerate(rooftop_detections.tracker_id):
        rooftop_id = int(rooftop_id)
        all_rooftops.add(rooftop_id)
        
        rooftop_box = rooftop_detections.xyxy[r_idx]
        has_solar = False
        
        # If solar panel detection exists
        if len(solar_detections.tracker_id) > 0:
            for s_idx, _ in enumerate(solar_detections.tracker_id):
                solar_box = solar_detections.xyxy[s_idx]
                # Calculate IoU using our custom function
                iou = calculate_iou(rooftop_box, solar_box)
                
                # If there's significant overlap, mark the rooftop as having solar panel
                if iou > 0.05:
                    has_solar = True
                    break
        
        # Update the rooftop status
        rooftop_status[rooftop_id] = has_solar
        
        # If this rooftop has solar, add it to our global tracking set
        if has_solar:
            rooftops_with_solar.add(rooftop_id)
    
    # Create custom labels for rooftops
    rooftop_labels = [
        f"#{tracker_id} Rooftop {'(Solar)' if rooftop_status[int(tracker_id)] else '(No Solar)'}"
        for tracker_id in rooftop_detections.tracker_id
    ]
    
    # Annotate rooftops
    annotated_frame = box_annotator.annotate(
        annotated_frame, 
        detections=rooftop_detections
    )
    
    # Add rooftop labels
    annotated_frame = label_annotator.annotate(
        annotated_frame, 
        detections=rooftop_detections, 
        labels=rooftop_labels
    )
    
    # Annotate solar panels with masks if available
    if hasattr(solar_results, 'masks') and solar_results.masks is not None:
        annotated_frame = mask_annotator.annotate(
            annotated_frame, 
            detections=solar_detections
        )
    
    
    # Add text showing statistics
    sv.ColorPalette.DEFAULT

    # Calculate the cumulative statistics for the entire video
    total_rooftops = len(all_rooftops)
    total_with_solar = len(rooftops_with_solar)
    total_without_solar = total_rooftops - total_with_solar
    
    # Define dimensions and padding - doubled from previous version
    stats_width = 500  
    stats_height = 240  
    padding = 20       
    row_height = 80    
    font_scale = 1.4   
    text_thickness = 3 
    
    # Create a black background for the stats
    cv2.rectangle(
        annotated_frame,
        (padding, padding),
        (padding + stats_width, padding + stats_height),
        (0, 0, 0),
        -1
    )
    
    # Add total rooftops text (white on black)
    cv2.rectangle(
        annotated_frame,
        (padding, padding),
        (padding + stats_width, padding + row_height),
        (0, 0, 0),
        -1
    )
    cv2.putText(
        annotated_frame,
        f"Total Rooftops: {total_rooftops}",
        (padding + 20, padding + 55),  # Adjusted for better centering
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness
    )
    
    # Add "With Solar" text (white on green)
    cv2.rectangle(
        annotated_frame,
        (padding, padding + row_height),
        (padding + stats_width, padding + 2*row_height),
        (87, 213, 59),  # Green in BGR format
        -1
    )
    cv2.putText(
        annotated_frame,
        f"With Solar: {total_with_solar}",
        (padding + 20, padding + row_height + 55),  # Adjusted for better centering
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness
    )
    
    # Add "Without Solar" text (white on red)
    cv2.rectangle(
        annotated_frame,
        (padding, padding + 2*row_height),
        (padding + stats_width, padding + 3*row_height),
        (59, 59, 213),  # Red in BGR format
        -1
    )
    cv2.putText(
        annotated_frame,
        f"Without Solar: {total_without_solar}",
        (padding + 20, padding + 2*row_height + 55),  # Adjusted for better centering
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        text_thickness
    )
    
    return annotated_frame


# Process the video
sv.process_video(
    source_path="solar_panel.mp4",
    target_path="rooftop_solar_analysis.mp4",
    callback=callback
)