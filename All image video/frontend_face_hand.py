# File: frontend_face_hand.py
import streamlit as st
import cv2
import mediapipe as mp

st.title("Real-time Face & Hand Tracking with MediaPipe")

use_webcam = st.checkbox("Use Webcam for Tracking")

if use_webcam:
    stframe = st.empty()

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

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

        # Draw face landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Webcam stopped.")
