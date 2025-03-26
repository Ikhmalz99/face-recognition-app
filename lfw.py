import os
import numpy as np
from itertools import combinations
from PIL import Image
import cv2
from deepface import DeepFace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mtcnn.mtcnn import MTCNN

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize face detector
detector = MTCNN()

# === Face preprocessing functions ===
def align_face(image):
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        return image[max(0, y):y + h, max(0, x):x + w]
    return None

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def augment_image(image):
    return np.fliplr(image)

# === Load and preprocess images ===
def load_images(dataset_path, limit_people=30, limit_images_per_person=2, augment=True):
    data = {}
    subdirs = sorted(os.listdir(dataset_path))[:limit_people]
    for subdir in subdirs:
        person_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(person_path):
            continue
        count = 0
        for file in os.listdir(person_path):
            if count >= limit_images_per_person:
                break
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    img_path = os.path.join(person_path, file)
                    img = Image.open(img_path).convert("RGB")
                    img = np.array(img)
                    face = align_face(img)
                    if face is None:
                        continue
                    face = cv2.resize(face, (224, 224))
                    face = normalize_image(face)

                    if subdir not in data:
                        data[subdir] = []

                    data[subdir].append((face, file))
                    count += 1

                    if augment:
                        data[subdir].append((augment_image(face), f"aug_{file}"))

                except Exception as e:
                    print(f" Failed to process {file}: {e}")
    return data

# === Evaluate the model with verification ===
def evaluate_model(data, threshold=0.6):
    y_true, y_pred = [], []

    print("\n Verifying SAME PERSON (True Matches):")
    for person, images in data.items():
        if len(images) >= 2:
            for (img1, name1), (img2, name2) in combinations(images, 2):
                try:
                    result = DeepFace.verify(img1, img2, model_name='VGG-Face', enforce_detection=False)
                    print(f"[True Match] {name1} vs {name2} → Match: {result['verified']} | Distance: {result['distance']:.4f}")
                    y_true.append(1)
                    y_pred.append(1 if result['distance'] < threshold else 0)
                except Exception as e:
                    print(f" Error verifying true pair: {e}")

    print("\n Verifying DIFFERENT PEOPLE (False Matches):")
    keys = list(data.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            try:
                img1, name1 = data[keys[i]][0]
                img2, name2 = data[keys[j]][0]

                result = DeepFace.verify(img1, img2, model_name='VGG-Face', enforce_detection=False)
                print(f"[False Match] {name1} vs {name2} → Match: {result['verified']} | Distance: {result['distance']:.4f}")
                y_true.append(0)
                y_pred.append(1 if result['distance'] < threshold else 0)
            except Exception as e:
                print(f" Error verifying false pair: {e}")

    if not y_true:
        print("No pairs evaluated.")
        return

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

# === Main Execution ===
if __name__ == "__main__":
    dataset_path = "C:/face_recognition/lfw_dataset/lfw-deepfunneled"
    print("Loading and preprocessing images...")
    data = load_images(dataset_path)
    print(f" Loaded {sum(len(v) for v in data.values())} preprocessed images across {len(data)} persons.")
    evaluate_model(data)
