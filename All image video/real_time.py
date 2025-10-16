# File: frontend_blazepose.py
import streamlit as st
import cv2
import mediapipe as mp

st.title("Real-time Body Pose Tracking with BlazePose")

# Checkbox to enable webcam
use_webcam = st.checkbox("Use Webcam for Pose Tracking")

if use_webcam:
    stframe = st.empty()

    # Initialize BlazePose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)  # webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        # Flip the frame for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Webcam stopped.")
