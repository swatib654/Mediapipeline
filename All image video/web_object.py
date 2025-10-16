# File: frontend_object_detection.py
import streamlit as st
import cv2
import mediapipe as mp
import tempfile

st.title("Object Detection with Webcam using MediaPipe")

# Streamlit checkbox for webcam
use_webcam = st.checkbox("Use Webcam")

if use_webcam:
    stframe = st.empty()

    # Initialize MediaPipe Holistic (pose, hands, face)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic()

    cap = cv2.VideoCapture(0)  # webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        # Flip frame for selfie view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Webcam stopped.")
