import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Load YOLOv10 model
model = YOLO('Trained_model.pt')


# Streamlit app
def main():
    st.title("Real-time Object Detection with YOLOv10")

    # Create a placeholder for the video feed
    video_placeholder = st.empty()

    # Start the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        #st.write(results)
        # Draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the annotated frame
        video_placeholder.image(annotated_frame, channels='BGR', use_column_width=True)

    cap.release()


if __name__ == "__main__":
    main()