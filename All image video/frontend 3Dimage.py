import streamlit as st
import plotly.graph_objects as go
import numpy as np
from PIL import Image

st.set_page_config(page_title="2D Image to 3D Viewer", layout="wide")
st.title("2D Image to 3D Viewer")

st.markdown("""
Upload any image (.jpg, .png) and visualize it as a simple 3D height map.
The brightness of pixels will determine the "height" in 3D.
""")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Load image
        img = Image.open(uploaded_file).convert("L")  # convert to grayscale
        img = img.resize((100, 100))  # resize for faster plotting
        data = np.array(img)

        # Create coordinates
        x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        z = data  # brightness as height

        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis")])
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Intensity"
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("Upload any image to visualize it in 3D.")




