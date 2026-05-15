import os
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = None
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None

class_labels = {
    0: "Bercak Daun",
    1: "Hawar Daun",
    2: "Karat Daun",
    3: "Sehat"
}

disease_info = {

    "Bercak Daun":
    """
    Penyakit bercak daun pada jagung disebabkan oleh infeksi jamur yang menyerang jaringan daun.
    Gejalanya berupa bercak kecil hingga besar berwarna coklat atau abu-abu yang dapat menyebar
    ke seluruh permukaan daun. Jika tidak ditangani, penyakit ini dapat menghambat proses
    fotosintesis sehingga pertumbuhan tanaman menjadi terganggu dan hasil panen menurun.

    <br><br>
    <b>Penanganan:</b>

    <ul>
        <li>Gunakan benih berkualitas dan tahan penyakit</li>
        <li>Jaga kelembapan lahan agar tidak terlalu tinggi</li>
        <li>Buang daun yang terinfeksi</li>
        <li>Gunakan fungisida sesuai anjuran</li>
    </ul>
    """,

    "Hawar Daun":
    """
    Hawar daun merupakan penyakit yang umum menyerang tanaman jagung akibat infeksi jamur.
    Penyakit ini ditandai dengan bercak memanjang berwarna coklat hingga abu-abu pada daun.
    Dalam kondisi lembap, bercak dapat meluas dengan cepat dan menyebabkan daun mengering.

    <br><br>
    <b>Dampak:</b>

    <ul>
        <li>Menurunkan kualitas fotosintesis</li>
        <li>Menghambat pertumbuhan tanaman</li>
        <li>Menurunkan produktivitas hasil panen</li>
    </ul>

    <br>
    <b>Penanganan:</b>

    <ul>
        <li>Gunakan varietas jagung tahan penyakit</li>
        <li>Hindari kelembapan berlebih</li>
        <li>Lakukan rotasi tanaman</li>
        <li>Gunakan fungisida secara tepat</li>
    </ul>
    """,

    "Karat Daun":
    """
    Karat daun disebabkan oleh jamur yang menimbulkan bintik kecil berwarna kuning kecoklatan
    menyerupai karat pada permukaan daun jagung. Penyakit ini mudah menyebar melalui angin
    dan lingkungan yang lembap.

    <br><br>
    <b>Gejala:</b>

    <ul>
        <li>Bintik kecil seperti serbuk karat</li>
        <li>Daun menjadi kering lebih cepat</li>
        <li>Pertumbuhan tanaman terganggu</li>
    </ul>

    <br>
    <b>Penanganan:</b>

    <ul>
        <li>Gunakan fungisida sesuai dosis</li>
        <li>Kurangi kelembapan area tanam</li>
        <li>Bersihkan sisa tanaman yang terinfeksi</li>
        <li>Gunakan varietas tahan penyakit</li>
    </ul>
    """,

    "Sehat":
    """
    Daun jagung berada dalam kondisi sehat dan tidak menunjukkan gejala penyakit.
    Warna daun tampak hijau segar, permukaan daun bersih, serta tidak terdapat bercak,
    karat, atau kerusakan jaringan daun.

    <br><br>
    <b>Saran Perawatan:</b>

    <ul>
        <li>Lakukan penyiraman secara teratur</li>
        <li>Gunakan pupuk sesuai kebutuhan tanaman</li>
        <li>Jaga kebersihan lahan</li>
        <li>Lakukan pemeriksaan rutin untuk mencegah penyakit</li>
    </ul>
    """
}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    probabilities = prediction[0] * 100

    predicted_index = int(np.argmax(prediction))
    predicted_label = class_labels[predicted_index]
    confidence = float(probabilities[predicted_index])

    probabilities_dict = {
        class_labels[i]: round(float(probabilities[i]), 2)
        for i in range(len(probabilities))
    }

    return predicted_label, round(confidence, 2), probabilities_dict


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            label, confidence, probabilities = predict_image(file_path)

            result = {
                "image_path": "/static/uploads/" + filename,
                "label": label,
                "confidence": confidence,
                "probabilities": probabilities,
                "info": disease_info.get(label, "-")
            }

    return render_template("index.html", result=result)


@app.route("/predict-live", methods=["POST"])
def predict_live():
    data = request.json["image"]
    image_data = data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    live_path = os.path.join(UPLOAD_FOLDER, "live_capture.jpg")

    with open(live_path, "wb") as f:
        f.write(image_bytes)

    label, confidence, probabilities = predict_image(live_path)

    return jsonify({
        "image_path": "/static/uploads/live_capture.jpg",
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "info": disease_info.get(label, "-")
    })


@app.route("/informasi")
def informasi():
    return render_template("informasi.html")


@app.route("/tentang")
def tentang():
    return render_template("tentang.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
