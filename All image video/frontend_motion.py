# File: frontend_motion_tracking.py
import streamlit as st
import cv2
import tempfile
import numpy as np

st.title("Motion Tracking App")

# Option to choose webcam or upload video
use_webcam = st.checkbox("Use Webcam for Motion Tracking")
uploaded_file = st.file_uploader("Or upload a video", type=["mp4", "avi", "mov"])

stframe = st.empty()

# Motion Tracking Function
def motion_tracking(video_source):
    cap = cv2.VideoCapture(video_source)
    
    ret, frame1 = cap.read()
    if not ret:
        st.warning("Cannot read video source.")
        return

    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Compute difference
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update previous frame
        prev_gray[:] = gray

        # Display frame in Streamlit
        frame_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

# Run motion tracking
if use_webcam:
    motion_tracking(0)  # webcam
elif uploaded_file is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    motion_tracking(tfile.name)
