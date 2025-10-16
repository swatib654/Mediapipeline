# File: frontend_hand_tracking.py
import streamlit as st
import cv2
import mediapipe as mp

st.title("Real-time Hand Tracking with MediaPipe")

# Checkbox to enable webcam
use_webcam = st.checkbox("Use Webcam for Hand Tracking")

if use_webcam:
    stframe = st.empty()

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)  # webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Cannot access webcam.")
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Webcam stopped.")
