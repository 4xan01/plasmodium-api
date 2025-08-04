from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model("modelo_parasitos.h5")

# Carpeta donde se guardan las imágenes cargadas
UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Ruta principal que acepta GET y POST
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'image' not in request.files:
            return "No se encontró la imagen"

        image = request.files['image']
        if image.filename == "":
            return "No se seleccionó ninguna imagen"

        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(image_path)

        # Procesamiento de imagen
        img = load_img(image_path, target_size=(150, 150))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]
        label = "✅ Parasitado" if prediction > 0.5 else "❌ No parasitado"
        label += f" ({prediction:.2f})"

        return render_template("index.html", result=label, filename=filename)

    return render_template("index.html", result=None, filename=None)


if __name__ == "__main__":
    app.run(debug=True)


