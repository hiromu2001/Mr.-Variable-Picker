import cv2
import time
import os
import numpy as np
from openvino.runtime import Core
from tracker import CentroidTracker
from analytics import RetailMetrics, CsvLogger

# ===== 1. モデルパス =====
MODEL_DIR = 'models'
# 顔検出はOpenCVのHaar Cascadeを使用
FACE_CASCADE_FILE = os.path.join(MODEL_DIR, 'haarcascade_frontalface_default.xml') 
AGE_GENDER_MODEL_XML = os.path.join(MODEL_DIR, 'age-gender-recognition-retail-0013.xml')
EMO_MODEL_XML = os.path.join(MODEL_DIR, 'emotions-recognition-retail-0003.xml')

EMO_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anger']
GENDER_LABELS = ['Female', 'Male']

# ===== 2. セッション生成と分類器の読み込み =====
# --- Haar Cascade 分類器 (顔検出用) ---
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)

# --- OpenVINO Runtime (年齢・性別・感情認識用) ---
core = Core()
age_gender_model = core.read_model(model=AGE_GENDER_MODEL_XML)
compiled_age_gender_model = core.compile_model(model=age_gender_model, device_name="CPU")
age_gender_input_layer = compiled_age_gender_model.input(0)
age_gender_output_layer_age = compiled_age_gender_model.output(0)
age_gender_output_layer_gender = compiled_age_gender_model.output(1)

emo_model = core.read_model(model=EMO_MODEL_XML)
compiled_emo_model = core.compile_model(model=emo_model, device_name="CPU")
emo_input_layer = compiled_emo_model.input(0)
emo_output_layer = compiled_emo_model.output(0)

# ===== 3. 推論ユーティリティ =====
# ▼▼▼ detect_faces関数をHaar Cascade用に完全に書き換え ▼▼▼
def detect_faces(img, face_cascade_classifier):
    # Haar Cascadeはグレースケール画像で動作
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 顔検出を実行
    faces_xywh = face_cascade_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(30, 30)
    )
    
    # trackerが要求する (x1, y1, x2, y2) の形式に変換
    results = []
    for (x, y, w, h) in faces_xywh:
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        results.append((x1, y1, x2, y2))
        
    return results
# ▲▲▲ ここまで ▲▲▲

def age_gender(crop):
    n, c, h, w = age_gender_input_layer.shape
    resized_crop = cv2.resize(crop, (w, h))
    inp = resized_crop.transpose(2, 0, 1)[np.newaxis, ...]
    results = compiled_age_gender_model([inp])
    age = int(results[age_gender_output_layer_age][0][0][0][0] * 100)
    gender_probs = results[age_gender_output_layer_gender][0][0]
    gender = GENDER_LABELS[np.argmax(gender_probs)]
    return age, gender

def emotion(crop):
    n, c, h, w = emo_input_layer.shape
    resized_crop = cv2.resize(crop, (w, h))
    inp = resized_crop.transpose(2, 0, 1)[np.newaxis, ...]
    results = compiled_emo_model([inp])[emo_output_layer]
    expr_idx = np.argmax(results[0])
    return EMO_LABELS[expr_idx]

# ===== 4. メインループ =====
tracker  = CentroidTracker()
metrics  = RetailMetrics()
cap      = cv2.VideoCapture(0)
with CsvLogger('logs') as logger:
    print('[q] キーで終了します')
    while True:
        ret, frame = cap.read()
        if not ret: break

        # detect_facesの呼び出し方を変更
        faces = detect_faces(frame, face_cascade)
        objects, deregistered_ids = tracker.update(faces)

        for obj_id in deregistered_ids:
            summary = metrics.get_person_summary(obj_id)
            if summary:
                logger.log(summary)
            metrics.finalize_person(obj_id)

        for obj_id, (x1, y1, x2, y2) in objects.items():
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            age, gender = age_gender(crop)
            expr        = emotion(crop)
            
            metrics.update(obj_id, expr, age, gender)
            
            stable_attrs = metrics.get_current_stable_attributes(obj_id)
            stable_age = stable_attrs.get("age", "?")
            stable_gender = stable_attrs.get("gender", "?")
            stable_expr = stable_attrs.get("expression", "?")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'ID:{obj_id} {stable_age} {stable_gender} {stable_expr}'
            
            if y1 < 30:
                y_pos = y1 + 15
            else:
                y_pos = y1 - 10
            cv2.putText(frame, label, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Retail Analytics', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print('CSV 保存先:', logger.path)
