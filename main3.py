import base64

import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np


def sidebar_bg(side_bg1):
    side_bg_ext = 'jpg'

    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg1, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )


side_bg = 'image.jpg'
sidebar_bg(side_bg)

# Load YOLOv10 model
model = YOLO('Trained_model.pt')

# Custom labels (replace with your desired labels)
custom_labels = {
    0: "Helmet",
    5: "No-Helmet",
    6: "Number Plate"
}


# Streamlit app
def main():
    st.title("Real-time Helmet Detection with YOLOv10")

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

        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()

        # Iterate through the detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(x2)

                # Get class and confidence
                cls = int(box.cls[0])
                conf = box.conf.item()

                # Get custom label
                label = custom_labels.get(cls, f"Class {cls}")

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put custom label and confidence score {conf:.2f}
                label_conf = f'{label}'
                cv2.putText(annotated_frame, label_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the annotated frame
        video_placeholder.image(annotated_frame, channels='BGR', use_column_width=True)

    cap.release()


if __name__ == "__main__":
    main()
