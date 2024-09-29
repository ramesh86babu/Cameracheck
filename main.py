import cv2
import os
import numpy as np
import streamlit as st



# Load the overlay image
overlay_image = cv2.imread('IOCL.jpg', cv2.IMREAD_UNCHANGED)

# Check if the overlay image was loaded successfully
if overlay_image is None:
    st.error("Error: Could not load overlay image. Check the file path and integrity.")
    exit()

# Function to overlay an image on a video frame
def overlay_image_on_frame(frame, overlay, position):
    x, y = position
    h, w = overlay.shape[:2]

    # Resize overlay if it's larger than the ROI
    if h > frame.shape[0] or w > frame.shape[1]:
        overlay = cv2.resize(overlay, (frame.shape[1] - x, frame.shape[0] - y))

    # Get the region of interest in the frame
    roi = frame[y:y+h, x:x+w]

    # Blend the images directly without checking for alpha
    combined = cv2.addWeighted(roi, 0.5, overlay, 0.5, 0)
    frame[y:y+h, x:x+w] = combined  # Set the blended image back to the frame

# Streamlit UI
st.title("Augmented Reality View with Brightness, Contrast, and Color Adjustment")

# Sliders for brightness, contrast, and color adjustment
brightness = st.slider("Adjust Brightness", -100, 100, 0, key="brightness_slider")
contrast = st.slider("Adjust Contrast", 0.0, 3.0, 1.0, step=0.1, key="contrast_slider")
color = st.slider("Adjust Color Balance (Hue)", -180, 180, 0, key="color_slider")

# Start capturing video from webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
FRAME_WINDOW = st.image([])

# Continuous video feed
while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video.")
        break

    # Reverse the video frame (flip horizontally)
    frame = cv2.flip(frame, 1)  # 1 for horizontal flip

    # Apply brightness adjustment
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)

    # Apply contrast adjustment
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)

    # Convert to HSV for color adjustment
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Adjust hue
    h = np.clip(h + color, 0, 179)  # Hue values in OpenCV are from 0 to 179
    hsv = cv2.merge((h, s, v))
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Define the position to overlay the image (top-left corner)
    position = (50, 50)  # Change this as needed

    # Overlay the image on the current frame
    overlay_image_on_frame(frame, overlay_image, position)

    # Convert the frame from BGR to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the resulting frame in Streamlit
    FRAME_WINDOW.image(frame, channels="RGB")

# Release the capture
cap.release()
st.success("AR View stopped.")
