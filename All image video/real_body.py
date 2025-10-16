# File: frontend_pose.py
import streamlit as st
import cv2
import mediapipe as mp
import tempfile

st.title("Real-time Body Pose Tracking")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    stframe = st.empty()
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert BGR to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()
    st.success("Processing finished!")
