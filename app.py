import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import cv2
import os

def draw_box_with_name(image_np, label, face_locations):
    img = image_np.copy()

    # Clean up label
    label = os.path.splitext(label)[0][:15]

    for top, right, bottom, left in face_locations:
        # Draw bounding box
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label box background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (left, top - text_height - 6), (left + text_width, top), (0, 255, 0), -1)
        cv2.putText(img, label, (left, top - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return img

# === Streamlit App ===
st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("Face Detection using face_recognition")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    # Load and convert
    image = face_recognition.load_image_file(uploaded_image)
    face_locations = face_recognition.face_locations(image)

    if face_locations:
        st.success(f"Found {len(face_locations)} face(s)")
        boxed_image = draw_box_with_name(image, os.path.basename(uploaded_image.name), face_locations)
        st.image(boxed_image, caption="Detected Faces", use_column_width=True)
    else:
        st.warning("No faces detected. Try another image.")
