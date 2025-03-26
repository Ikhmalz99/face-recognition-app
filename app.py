import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import os

# === Cached model loader ===
@st.cache_resource
def load_model(model_name="Facenet"):
    return DeepFace.build_model(model_name)

model_name = "Facenet"  # You can switch to "VGG-Face" or others
model = load_model(model_name)

# === Preprocessing functions ===
def align_face(image_np):
    detector = MTCNN()
    faces = detector.detect_faces(image_np)
    if faces:
        x, y, w, h = faces[0]['box']
        return image_np[max(0, y):y + h, max(0, x):x + w]
    return None

def normalize_image(image_np):
    image_np = cv2.resize(image_np, (224, 224))
    return image_np.astype(np.float32) / 255.0

def draw_box_with_name(image_np, label, max_faces=1):
    detector = MTCNN()
    faces = detector.detect_faces(image_np)
    img = image_np.copy()
    label = os.path.splitext(label)[0][:15]

    for i, face in enumerate(faces[:max_faces]):
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img, (x, y - text_height - 6), (x + text_width, y), (0, 255, 0), -1)
        cv2.putText(img, label, (x, y - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return img

# === Streamlit App ===
st.set_page_config(page_title="LFW Face Verification", layout="centered")
st.title("Face Recognition using LFW Preprocessing + Facenet")

img1_file = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("RGB")
    img2 = Image.open(img2_file).convert("RGB")

    st.image([img1, img2], caption=["Image 1", "Image 2"], width=250)

    img1_np = np.array(img1)
    img2_np = np.array(img2)

    aligned1 = align_face(img1_np)
    aligned2 = align_face(img2_np)

    if aligned1 is None or aligned2 is None:
        st.error("Face not detected in one of the images. Please try again with clearer images.")
    else:
        norm1 = normalize_image(aligned1)
        norm2 = normalize_image(aligned2)

        with st.spinner("Verifying faces..."):
            result = DeepFace.verify(
                img1_path=norm1,
                img2_path=norm2,
                model_name=model_name,
                model=model,
                enforce_detection=False
            )

        st.success("Verification Result")
        st.write(f"**Match:** {result['verified']}")
        st.write(f"**Distance:** {result['distance']:.4f}")

        boxed1 = draw_box_with_name(img1_np, os.path.basename(img1_file.name))
        boxed2 = draw_box_with_name(img2_np, os.path.basename(img2_file.name))

        st.subheader("Face Detection with Labels")
        st.image([boxed1, boxed2], caption=["Image 1", "Image 2"], width=300)
