import streamlit as st
import cv2
import mediapipe as mp
import tempfile
from PIL import Image
import numpy as np

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.set_page_config(page_title="Pose Landmark Detection", layout="centered")

st.title("🧍 Pose Landmark Detection using MediaPipe")
st.write("Upload an image or use your webcam to detect human pose landmarks in real time.")

# Sidebar options
option = st.sidebar.radio("Choose Input Type:", ("Upload Image", "Use Webcam"))

# Pose detection function
def detect_pose(image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )
        return annotated_image

# Upload Image
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result = detect_pose(image)
        if result is not None:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Pose Landmarks", use_container_width=True)
        else:
            st.warning("No pose landmarks detected.")

# Webcam mode
elif option == "Use Webcam":
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("Camera not available.")
                break
            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        camera.release()
