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

# Initialize MTCNN detector
detector = MTCNN()

# Preprocessing function (align + resize + normalize)
def preprocess_face(image):
    faces = detector.detect_faces(image)
    if faces:
        x, y, w, h = faces[0]['box']
        face = image[max(0, y):y + h, max(0, x):x + w]
        face = cv2.resize(face, (224, 224))
        face = face.astype(np.float32) / 255.0
        return face
    return None

# Augmentation function (flip horizontally)
def augment_image(image):
    return np.fliplr(image)

# Load images from dataset and apply preprocessing/augmentation
def load_images(dataset_path, people_limit=30, images_per_person=2, augment=True):
    data = {}
    subfolders = sorted(os.listdir(dataset_path))[:people_limit]

    for person in subfolders:
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue

        image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images = []

        for file in image_files[:images_per_person]:
            try:
                img_path = os.path.join(person_path, file)
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                face = preprocess_face(img_np)

                if face is not None:
                    images.append((face, file))
                    if augment:
                        aug_face = augment_image(face)
                        images.append((aug_face, f"aug_{file}"))

            except Exception as e:
                print(f"Error processing {file}: {e}")

        if images:
            data[person] = images

    return data

# Model Evaluation 
def evaluate_model(data, threshold=0.3):
    y_true = []
    y_pred = []

    # Check same person pairs (true matches)
    print("\n--- SAME PERSON VERIFICATION ---")
    for person, images in data.items():
        if len(images) < 2:
            continue
        for (img1, name1), (img2, name2) in combinations(images, 2):
            try:
                result = DeepFace.verify(img1, img2, model_name='VGG-Face', enforce_detection=False)
                distance = result['distance']
                is_match = distance < threshold
                print(f"[True] {name1} vs {name2} → Distance: {distance:.4f} → Match: {is_match}")
                y_true.append(1)
                y_pred.append(1 if is_match else 0)
            except Exception as e:
                print(f"Error verifying true pair: {e}")

    # Check different person pairs (false matches)
    print("\n--- DIFFERENT PEOPLE VERIFICATION ---")
    people = list(data.keys())
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            try:
                img1, name1 = data[people[i]][0]
                img2, name2 = data[people[j]][0]
                result = DeepFace.verify(img1, img2, model_name='VGG-Face', enforce_detection=False)
                distance = result['distance']
                is_match = distance < threshold
                print(f"[False] {name1} vs {name2} → Distance: {distance:.4f} → Match: {is_match}")
                y_true.append(0)
                y_pred.append(1 if is_match else 0)
            except Exception as e:
                print(f"Error verifying false pair: {e}")

    # Print metrics
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print("\n--- EVALUATION RESULTS ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1 Score : {f1:.4f}")
    else:
        print("No image pairs evaluated.")

# Main function
if __name__ == "__main__":
    dataset_path = "C:/face_recognition/lfw_dataset/lfw-deepfunneled"
    print("Loading dataset...")
    data = load_images(dataset_path, people_limit=30, images_per_person=2, augment=True)
    print(f"Loaded {sum(len(v) for v in data.values())} images across {len(data)} people.")
    evaluate_model(data)
