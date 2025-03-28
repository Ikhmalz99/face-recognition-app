# face-recognition-app
Face recognition system using DeepFace and Streamlit

--- 

# Face Recognition App using DeepFace and Streamlit

This project demonstrates a complete face recognition system using the **LFW dataset**, **DeepFace (VGG-Face model)**, and a **Streamlit GUI**. It includes:

- Face **preprocessing** (alignment, normalization, augmentation)
- Model **evaluation** using accuracy, precision, recall, F1-score
- **Live face verification** GUI with bounding boxes and labels
- Deployment-ready code for **Streamlit Cloud**

---

## Project Structure

```
face-recognition-app/
├── app.py             # Streamlit GUI
├── lfw.py             # Preprocessing and evaluation
├── requirements.txt   # Dependencies (DeepFace, Streamlit, etc.)
├── packages.txt       # System-level packages (e.g. distutils, libGL)
├── runtime.txt        # Python version pin (3.10.13)
└── README.md          # Project instructions
```

---

## Installation

Make sure Python **3.8+** is installed. Then run:

```bash
# Clone the repository (optional if running locally)
git clone https://github.com/ikhmalz99/face-recognition-app.git
cd face-recognition-app

# Install dependencies
pip install -r requirements.txt
```

---

## Run Locally

Start the Streamlit app:

```bash
streamlit run app.py
```

---

## Model Evaluation (lfw.py)
The lfw.py script loads images from the LFW dataset, applies preprocessing (face alignment, normalization, and optional augmentation), and evaluates the model using the VGG-Face model from DeepFace.

It computes:

- Cosine distances between image pairs
- Accuracy, Precision, Recall, F1-score
- Threshold is customizable (default = 0.3).

---

## Evaluation Sample Output
The lfw.py script evaluates face recognition performance across pairs of images using VGG-Face:

--- EVALUATION RESULTS ---
Accuracy : 0.9709
Precision: 0.8736
Recall   : 0.9500
F1 Score : 0.9102

---

## GUI Features (app.py)
The Streamlit GUI allows:

- Uploading two face images
- Face detection with bounding boxes
- Normalization + horizontal flip augmentation
- Face verification using DeepFace.verify() (average of original and augmented comparison)
- Displays:
1. Match or Not
2. Distance scores
3. Detected faces with name label overlay

---

## Deployment (Streamlit Cloud)
You can easily deploy this app using Streamlit Cloud.

Steps:
1. Push your code to GitHub
Ensure the following files are included:

- app.py
- lfw.py
- requirements.txt
- packages.txt
- runtime.txt
- README.md

2. Visit Streamlit Cloud and sign in with GitHub.

3. Click “New App”
- Choose your repo, branch (usually main), and app.py as the main file.

4. Click “Deploy”
- Streamlit will automatically install the dependencies and launch your app.

Note: If you're using large datasets (like LFW), avoid uploading the whole dataset. Instead, upload your own images during runtime, or store datasets externally (e.g., Google Drive or a cloud bucket).

---

## Demo Screenshot

Below is a preview of the face verification GUI with bounding boxes and labels:

![Demo](https://github.com/ikhmalz99/face-recognition-app/raw/main/demo_screenshot.png)

