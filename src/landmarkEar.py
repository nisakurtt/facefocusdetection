import os
import cv2
import mediapipe as mp
import math
import numpy as np
import time

# EAR eşik değeri (deneysel, genelde 0.20–0.25 arası iyi çalışır)
EAR_THRESHOLD = 0.23
# Göz açıkken EAR genelde daha büyük çıkar. EAR>= threshold ise göz açıktır.

# Kayıt aralığı (her kaç saniyede 1 frame kaydedilsin)
SAVE_INTERVAL = 0.5

# Dataset yolları
# Verileri otomatik olarak 3 sınıfa ayıracağız.
BASE_DIR = "dataset"
OPEN_DIR = os.path.join(BASE_DIR, "open_eye")
CLOSED_DIR = os.path.join(BASE_DIR, "closed_eye")
UNFOCUSED_DIR = os.path.join(BASE_DIR, "unfocused")

# Klasörleri otomatik oluştur
for path in [OPEN_DIR, CLOSED_DIR, UNFOCUSED_DIR]:
    os.makedirs(path, exist_ok=True)  # klasör zaten varsa hata verme

# EAR değerine göre etiket belirleme fonksiyonu
def label_eye_state(avg_ear, threshold=EAR_THRESHOLD):
    if avg_ear >= threshold:
        return "open"
    else:
        return "closed"

# Frame kaydetme fonksiyonu
def save_frame(frame, label):
    timestamp = time.strftime("%Y%m%d_%H%M%S_%f")  # (not: %f time.strftime'de gerçek mikrosaniye değildir)
    if label == "open":
        path = os.path.join(OPEN_DIR, f"{timestamp}.jpg")
    elif label == "closed":
        path = os.path.join(CLOSED_DIR, f"{timestamp}.jpg")
    else:  # unfocused
        path = os.path.join(UNFOCUSED_DIR, f"{timestamp}.jpg")

    cv2.imwrite(path, frame)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Göz Noktaları Mediapipe Haritası
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

def calculate_distance(p1, p2):
    x_fark = p1.x - p2.x
    y_fark = p1.y - p2.y
    return math.sqrt(x_fark**2 + y_fark**2)

def get_eye_landmarks(face_landmarks, eye_indices):
    eye_points = []
    for idx in eye_indices:
        point = face_landmarks.landmark[idx]
        eye_points.append(point)
    return eye_points

def calculate_ear_formula(eye_points):
    dikey1 = calculate_distance(eye_points[1], eye_points[5])  # P2 - P6
    dikey2 = calculate_distance(eye_points[2], eye_points[4])  # P3 - P5
    yatay = calculate_distance(eye_points[0], eye_points[3])   # P1 - P4
    if yatay == 0:
        return 0
    ear = (dikey1 + dikey2) / (2.0 * yatay)
    return ear

def extract_eye_roi(frame, eye_landmarks, image_w, image_h, padding=10):
    pixel_points = []
    for lm in eye_landmarks:
        px = int(lm.x * image_w)
        py = int(lm.y * image_h)
        pixel_points.append([px, py])

    pixel_points = np.array(pixel_points)
    x, y, w, h = cv2.boundingRect(pixel_points)

    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image_w - x)
    h = min(h + 2 * padding, image_h - y)

    roi_image = frame[y:y+h, x:x+w]
    return roi_image

def prepare_for_cnn(roi_image, target_size=(64, 64)):
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 1, target_size[0], target_size[1]))
    return reshaped

# modeli başlat
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

last_save_time = 0

# Kameradan görüntü al ve yüzleri tespit et
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image_h, image_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_time = time.time()

    # yüz VARSA
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_points = get_eye_landmarks(face_landmarks, LEFT_EYE)
            right_points = get_eye_landmarks(face_landmarks, RIGHT_EYE)

            left_ear = calculate_ear_formula(left_points)
            right_ear = calculate_ear_formula(right_points)
            avg_ear = (left_ear + right_ear) / 2.0

            # Etiket üret (avg_ear hesaplandıktan sonra!)
            label = label_eye_state(avg_ear)

            cv2.putText(frame, f"Label: {label.upper()}",
                        (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            # belirli aralıklarla kaydet
            if current_time - last_save_time >= SAVE_INTERVAL:
                save_frame(frame, label)
                last_save_time = current_time

            # Sol göz ROI
            left_roi = extract_eye_roi(frame, left_points, image_w, image_h)
            if left_roi.size > 0:
                left_roi_show = cv2.resize(left_roi, (200, 100))
                cv2.imshow('Sol göz ROI', left_roi_show)

            # Sağ göz ROI
            right_roi = extract_eye_roi(frame, right_points, image_w, image_h)
            if right_roi.size > 0:
                right_roi_show = cv2.resize(right_roi, (200, 100))
                cv2.imshow('Sağ göz ROI', right_roi_show)

            # CNN input hazırlama (ROI boşsa burada hata verebilir; istersen ekstra kontrol ekleyebilirim)
            left_input = prepare_for_cnn(left_roi)
            right_input = prepare_for_cnn(right_roi)

            # EAR değerlerini ekrana yaz
            cv2.putText(frame, f'Sol EAR: {left_ear:.2f}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Sag EAR: {right_ear:.2f}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Ortalama EAR: {avg_ear:.2f}', (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # yüz YOKSA -> ODAKSIZ
    else:
        cv2.putText(frame, "Label: UNFOCUSED",
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        if current_time - last_save_time >= SAVE_INTERVAL:
            save_frame(frame, "unfocused")
            last_save_time = current_time

    cv2.imshow('MediaPipe FaceMesh', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
