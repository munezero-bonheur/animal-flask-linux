import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template_string, request
from PIL import Image, UnidentifiedImageError

folder = Path(__file__).resolve().parent
model = tf.keras.models.load_model(folder / "animal_cnn.keras")
classes = json.loads((folder / "animal_classes.json").read_text())

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

PAGE = """
<h1>Animal classifier</h1>
<p>Choose a photo of a Butterfly, Lizard, Spider, Frog, or Horse.</p>
<form method="post" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*" required>
  <button type="submit">Classify</button>
</form>
{% if result %}
  <h2>Prediction: {{ result }}</h2>
  <p>Known-class score: {{ score }}% (not a measure of confidence for unfamiliar animals).</p>
{% endif %}
{% if error %}<p>{{ error }}</p>{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE)

    if "image" not in request.files:
        return render_template_string(PAGE, error="Choose an image first."), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, OSError, ValueError):
        return render_template_string(PAGE, error="Could not read that image."), 400

    # The saved CNN already scales pixels internally: keep values in 0–255.
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    scores = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(scores))
    return render_template_string(
        PAGE, result=classes[best], score=round(float(scores[best]) * 100, 1)
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

