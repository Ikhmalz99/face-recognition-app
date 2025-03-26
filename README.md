# face-recognition-app
Face recognition system using DeepFace and Streamlit

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
├── app.py            # Streamlit GUI for face verification
├── lfw.py            # Model training, preprocessing, evaluation
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```
## Installation

Make sure Python **3.8+** is installed.

```bash
pip install -r requirements.txt

Run Locally
Start the Streamlit app:
streamlit run app.py

```
## Deployment (Streamlit Cloud)
You can easily deploy this app using Streamlit Cloud.

Steps:
1. Push your code to GitHub
Ensure the following files are included:

- app.py
- lfw.py
- requirements.txt
- README.md

2. Visit Streamlit Cloud and sign in with GitHub.

3. Click “New App”
- Choose your repo, branch (usually main), and app.py as the main file.

4. Click “Deploy”
- Streamlit will automatically install the dependencies and launch your app.

Note: If you're using large datasets (like LFW), avoid uploading the whole dataset. Instead, upload your own images during runtime, or store datasets externally (e.g., Google Drive or a cloud bucket).
