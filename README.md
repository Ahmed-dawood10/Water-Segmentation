# 🌊 Water Segmentation using U-Net

This project implements a **U-Net based deep learning model** for **water body segmentation** from satellite images.  
It provides a Flask web interface where users can upload a `.tif` image and get the predicted **water mask** instantly.

---

## 🚀 Features
- Upload `.tif` satellite images.
- Predict water body areas using a trained **U-Net model**.
- Display original image and predicted **segmentation mask** side by side.
- User-friendly frontend (HTML + CSS in Flask templates).
- Organized project structure for easy deployment.

---

## 📂 Project Structure
``` bash
SATELLITE-WATER-SEGMENTATION/
│── app.py # Main Flask app
│── pretrained_u_net.h5 # Pretrained U-Net model (not uploaded)
│── pretrained_u_net.ipynb # Notebook for pretrained model
│── u-net_fromscratch.h5 # From-scratch trained model (not uploaded)
│── u-net_fromscratch.ipynb # Notebook for from-scratch model
│── requirements.txt # Python dependencies
│── .gitignore # Ignore data + model files
│── README.md # Documentation
│
├── data/ # Dataset (ignored in Git)
│ ├── images/
│ └── labels/
│
├── uploads/ # Uploaded .tif files
│
├── static/ # Static assets
│ ├── style.css
│ └── results/ # Saved prediction masks
│
└── templates/ # HTML templates
└── index.html

```

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Ahmed-dawood10/Water-Segmentation.git
cd water-segmentation
```
### 2️⃣ Create and activate a Conda environment

conda create -n water-seg python=3.9 -y

conda activate water-seg

### 3️⃣ Install dependencies

pip install -r requirements.txt

### ▶️ Usage

Run the Flask app:

python app.py

Then open your browser at:

http://127.0.0.1:5000

Upload a .tif image, and the model will generate a segmentation mask highlighting water regions.

## 📦 Requirements
 Main dependencies (see requirements.txt for full list):

Python 3.9+

Flask

TensorFlow / Keras

NumPy

Pillow

Rasterio

Matplotlib


## 📊 Models

This project supports two U-Net models:

Pretrained U-Net (pretrained_u_net.h5)

From-Scratch U-Net (u-net_fromscratch.h5)

📌 Both models are too large to include in GitHub.
👉 You can download them from Google Drive:

Pretrained U-Net ([Drive Link](https://drive.google.com/file/d/1Zeg6qpXruHVJK_odAQyOpxoPxzuHqMtv/view?usp=drive_link))

From-Scratch U-Net (Drive Link)









