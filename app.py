import os
import base64
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO


app = Flask(__name__, static_folder=".")
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


print("Loading Keras Model: canteen_cnn_final.keras")
keras_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "canteen_cnn_final.keras"))
print("Keras Model loaded successfully.")


print("Loading YOLO Model: best.pt")
yolo_model = YOLO(os.path.join(BASE_DIR, "best.pt"))
print("YOLO Model loaded successfully.")


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


def get_crop_coordinates(w, h):
   """
   Trả về tọa độ 5 ngăn trên khay ăn inox chuẩn (x1, y1, x2, y2).
   Hàng trên: 2 ngăn lớn (Cơm, Canh)
   Hàng dưới: 3 ngăn nhỏ (Món mặn 1, Món mặn 2, Món xào/rau)
   """
   return [
       {"name": "Ngăn 1", "box": (int(372 / 1920 * w), int(122 / 1080 * h), int(880 / 1920 * w), int(584 / 1080 * h))},
       {"name": "Ngăn 2", "box": (int(1074 / 1920 * w), int(128 / 1080 * h), int(1452 / 1920 * w), int(590 / 1080 * h))},
       {"name": "Ngăn 3", "box": (int(370 / 1920 * w), int(672 / 1080 * h), int(684 / 1920 * w), int(934 / 1080 * h))},
       {"name": "Ngăn 4", "box": (int(738 / 1920 * w), int(672 / 1080 * h), int(1078 / 1920 * w), int(934 / 1080 * h))},
       {"name": "Ngăn 5", "box": (int(1124 / 1920 * w), int(674 / 1080 * h), int(1452 / 1920 * w), int(938 / 1080 * h))},
   ]


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
       (255, 165, 0),   # Cam (Rice)
       (0, 191, 255),   # Xanh dương (Soup)
       (50, 205, 50),   # Xanh lá (Dish 1)
       (238, 130, 238), # Tím (Dish 2)
       (255, 69, 0)     # Đỏ cam (Dish 3)
   ]


   for idx, item in enumerate(crops_info):
       x1, y1, x2, y2 = item["box"]
      
       # Cắt ảnh ngăn đồ ăn
       crop = img[y1:y2, x1:x2]
       if crop.size == 0:
           continue
          
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
      
       note = f"conf={confidence:.2f}"
       egg_count = 0
      
       if dish_class == "thit_kho_trung":
           yolo_res = yolo_model(crop)
           # YOLO model phát hiện 'egg half' hoặc 'egg whole'
           detected_eggs = len(yolo_res[0].boxes)
           egg_count = max(1, detected_eggs) # Mặc định có ít nhất 1 quả trứng
          
           extra_eggs = max(0, egg_count - 1)
           price += extra_eggs * 6000
           note += f", trứng={egg_count}"
           ui_name = f"Thịt kho trứng (x{egg_count} quả)"
          
           for box in yolo_res[0].boxes:
               bx1, by1, bx2, by2 = map(int, box.xyxy[0])
               ex1, ey1 = x1 + bx1, y1 + by1
               ex2, ey2 = x1 + bx2, y1 + by2
               cv2.rectangle(img_annotated, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)
               cv2.putText(img_annotated, "trung", (ex1, ey1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


       results.append({
           "id": ui_id,
           "name": ui_name,
           "price": price,
           "confidence": confidence,
           "slot": idx + 1
       })
      
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



