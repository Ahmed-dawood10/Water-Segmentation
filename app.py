import os
import numpy as np
import rasterio
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
from matplotlib.image import imsave
import uuid

# -------- Custom Loss & Metrics --------
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# -------- Load Model --------
model = load_model(
    "u-net_fromscratch.h5",
    custom_objects={"iou_metric": iou_metric,
                    "dice_loss": dice_loss,
                    "bce_dice_loss": bce_dice_loss}
)
print("Model loaded successfully.")

# -------- Flask App --------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    mask_filename = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Read TIFF
        with rasterio.open(filepath) as src:
            img = src.read()

        img_norm = np.zeros_like(img, dtype=np.float32)
        for b in range(img.shape[0]):
            band = img[b]
            img_norm[b] = (band - band.min()) / (band.max() - band.min() + 1e-8)

        img_norm = np.transpose(img_norm, (1, 2, 0))
        img_norm = np.expand_dims(img_norm, axis=0)

        # Predict
        pred = model.predict(img_norm)[0, ..., 0]
        pred_bin = (pred > 0.5).astype(np.uint8)

        # Save mask as PNG
        result_name = f"mask_{uuid.uuid4().hex}.png"
        result_path = os.path.join("static", "results", result_name)
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        imsave(result_path, (pred_bin * 255).astype(np.uint8))

        mask_filename = result_name  # اسم الملف فقط

    return render_template("index.html", mask_filename=mask_filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
