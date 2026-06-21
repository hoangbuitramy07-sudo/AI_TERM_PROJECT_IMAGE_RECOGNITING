import os
import base64
import json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EGG_COUNT_MODEL_PATH = os.path.join(BASE_DIR, "egg_count_cnn.keras")
EGG_COUNT_LABELS_PATH = os.path.join(BASE_DIR, "egg_count_labels.json")

print("Loading Keras Model: canteen_cnn_final.keras...")
keras_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "canteen_cnn_final.keras"))
print("Keras Model loaded successfully.")

egg_count_model = None
egg_count_labels = [1, 2, 3, 4]

if os.path.exists(EGG_COUNT_MODEL_PATH):
    print("Loading Egg Count CNN Model: egg_count_cnn.keras...")
    egg_count_model = tf.keras.models.load_model(EGG_COUNT_MODEL_PATH)
    if os.path.exists(EGG_COUNT_LABELS_PATH):
        with open(EGG_COUNT_LABELS_PATH, "r", encoding="utf-8") as f:
            egg_count_labels = [int(label) for label in json.load(f)]
    print("Egg Count CNN Model loaded successfully.")
else:
    print("WARNING: egg_count_cnn.keras not found. Thịt kho trứng will default to 1 egg.")

CANONICAL_CLASSES = [
    "com_trang",
    "dau_hu_sot_ca",
    "ca_hu_kho",
    "thit_kho_trung",
    "thit_kho",
    "canh_chua_co_ca",
    "canh_chua_khong_ca",
    "suon_nuong",
    "canh_rau_cai_thao",
    "canh_rau_muong",
    "rau_xao_cu_san",
    "rau_xao_dau_dua",
    "rau_xao_dau_que",
    "rau_xao_lagim",
    "trung_chien",
    "trung_chien_thit",
]

CLASS_TO_UI_ID = {
    "com_trang": "com",
    "dau_hu_sot_ca": "dau_hu_sot_ca",
    "ca_hu_kho": "ca_hu_kho",
    "thit_kho_trung": "thit_kho_trung",
    "thit_kho": "thit_kho",
    "canh_chua_co_ca": "canh_chua_co_ca",
    "canh_chua_khong_ca": "canh_chua_khong_ca",
    "suon_nuong": "suon_nuong",
    "canh_rau_cai_thao": "canh_rau_cai_thao",
    "canh_rau_muong": "canh_rau_muong",
    "rau_xao_cu_san": "rau_xao_cu_san",
    "rau_xao_dau_dua": "rau_xao_dau_dua",
    "rau_xao_dau_que": "rau_xao_dau_que",
    "rau_xao_lagim": "rau_xao_lagim",
    "trung_chien": "trung_chien",
    "trung_chien_thit": "trung_chien_thit",
}

CLASS_TO_UI_NAME = {
    "com_trang": "Cơm trắng",
    "dau_hu_sot_ca": "Đậu hũ sốt cà",
    "ca_hu_kho": "Cá hú kho",
    "thit_kho_trung": "Thịt kho trứng",
    "thit_kho": "Thịt kho",
    "canh_chua_co_ca": "Canh chua có cá",
    "canh_chua_khong_ca": "Canh chua không cá",
    "suon_nuong": "Sườn nướng",
    "canh_rau_cai_thao": "Canh rau cải thảo",
    "canh_rau_muong": "Canh rau muống",
    "rau_xao_cu_san": "Rau xào củ sắn",
    "rau_xao_dau_dua": "Rau xào đậu đũa",
    "rau_xao_dau_que": "Rau xào đậu que",
    "rau_xao_lagim": "Rau xào la gim",
    "trung_chien": "Trứng chiên",
    "trung_chien_thit": "Trứng chiên thịt",
}

PRICE_TABLE = {
    "com_trang": 10000,
    "dau_hu_sot_ca": 25000,
    "ca_hu_kho": 30000,
    "thit_kho_trung": 30000,
    "thit_kho": 25000,
    "canh_chua_co_ca": 25000,
    "canh_chua_khong_ca": 10000,
    "suon_nuong": 30000,
    "canh_rau_cai_thao": 7000,
    "canh_rau_muong": 7000,
    "rau_xao_cu_san": 10000,
    "rau_xao_dau_dua": 10000,
    "rau_xao_dau_que": 10000,
    "rau_xao_lagim": 10000,
    "trung_chien": 25000,
    "trung_chien_thit": 25000,
}

def predict_egg_count(crop):
    if egg_count_model is None:
        return 1, 0.0

    crop_resized = cv2.resize(crop, (192, 192))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    crop_batch = np.expand_dims(crop_rgb, axis=0).astype(np.float32)

    preds = egg_count_model.predict(crop_batch, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    if class_idx >= len(egg_count_labels):
        return 1, confidence

    return max(1, int(egg_count_labels[class_idx])), confidence

def get_crop_coordinates(w, h):
    return [
        {"name": "Ngăn 1", "box": (int(372 / 1920 * w), int(122 / 1080 * h), int(880 / 1920 * w), int(584 / 1080 * h))},
        {"name": "Ngăn 2", "box": (int(1074 / 1920 * w), int(128 / 1080 * h), int(1452 / 1920 * w), int(590 / 1080 * h))},
        {"name": "Ngăn 3", "box": (int(370 / 1920 * w), int(672 / 1080 * h), int(684 / 1920 * w), int(934 / 1080 * h))},
        {"name": "Ngăn 4", "box": (int(738 / 1920 * w), int(672 / 1080 * h), int(1078 / 1920 * w), int(934 / 1080 * h))},
        {"name": "Ngăn 5", "box": (int(1124 / 1920 * w), int(674 / 1080 * h), int(1452 / 1920 * w), int(938 / 1080 * h))},
    ]

def predict_dish(crop, slot=1):
    crop_resized = cv2.resize(crop, (192, 192))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    crop_batch = np.expand_dims(crop_rgb, axis=0).astype(np.float32)

    preds = keras_model.predict(crop_batch, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    dish_class = CANONICAL_CLASSES[class_idx]

    ui_id = CLASS_TO_UI_ID.get(dish_class, "")
    ui_name = CLASS_TO_UI_NAME.get(dish_class, "Chưa nhận diện")
    price = PRICE_TABLE.get(dish_class, 0)

    if dish_class == "thit_kho_trung":
        egg_count, egg_confidence = predict_egg_count(crop)
        extra_eggs = max(0, egg_count - 1)
        price += extra_eggs * 6000
        ui_name = f"Thịt kho trứng (x{egg_count} quả)"

    return {
        "id": ui_id,
        "name": ui_name,
        "price": price,
        "confidence": confidence,
        "slot": slot
    }

@app.route("/")
def index():
    return send_from_directory(".", "ui.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400

    h, w, c = img.shape
    crops_info = get_crop_coordinates(w, h)
    
    results = []
    img_annotated = img.copy()

    colors = [
        (255, 165, 0),
        (0, 191, 255),
        (50, 205, 50),
        (238, 130, 238),
        (255, 69, 0)
    ]

    for idx, item in enumerate(crops_info):
        x1, y1, x2, y2 = item["box"]
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        results.append(predict_dish(crop, idx + 1))

        color = colors[idx % len(colors)]
        cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 3)
        
    _, buffer = cv2.imencode(".jpg", img_annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    img_data_url = f"data:image/jpeg;base64,{img_base64}"
    
    return jsonify({
        "predictions": results,
        "annotated_image": img_data_url
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
