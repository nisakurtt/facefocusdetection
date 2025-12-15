import os
import cv2
import mediapipe as mp
import math
import numpy as np
from datetime import datetime
import time
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils 
mp_face_mesh = mp.solutions.face_mesh #yüz modülü

# EAR eşik değeri (deneysel, genelde 0.20–0.25 arası iyi çalışır)
EAR_THRESHOLD = 0.23
#Göz açıkken EAR genelde daha büyük çıkar. EAR>= threshold ise göz açıktır.

# Kayıt aralığı (her kaç saniyede 1 frame kaydedilsin)
SAVE_INTERVAL = 0.5  

# Dataset yolları
#Verileri otomatik olarak 3 sınıfa ayıracağız.
BASE_DIR = "dataset"   
OPEN_DIR = os.path.join(BASE_DIR, "open_eye")
CLOSED_DIR = os.path.join(BASE_DIR, "closed_eye")
UNFOCUSED_DIR = os.path.join(BASE_DIR, "unfocused")

# Klasörleri otomatik oluştur
for path in [OPEN_DIR, CLOSED_DIR, UNFOCUSED_DIR]:
    os.makedirs(path, exist_ok=True)   # exist_ok=True -> klasör zaten varsa hata verme devam et demek
 

#EAR değerine göre etiket belirleme fonksiyonu  
def label_eye_state(avg_ear, threshold=EAR_THRESHOLD):
    
    if avg_ear >= threshold:
        return "open"
    else:
        return "closed"


# Frame kaydetme fonksiyonu
def save_frame(frame, label):  #Frame' i labele göre uygun klasöre kaydeder.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  #Anlık tarih + saat + milisaniye
    #timestamp kullandık çünkü aynı isimle dosya çakışmasını önlüyor.
    #datasetin kronolojik ve düzenli şekilde oluşmasını sağlıyor.
    if label == "open":
        path = os.path.join(OPEN_DIR, f"{timestamp}.jpg") #İşletim sistemine uygun dosya yolu oluşturur.
    elif label == "closed":
        path = os.path.join(CLOSED_DIR, f"{timestamp}.jpg")
    else:  # unfocused
        path = os.path.join(UNFOCUSED_DIR, f"{timestamp}.jpg")
    
    #Frame 'i diske kaydetme 
    cv2.imwrite(path, frame)  #Kameradan gelen görüntüyü JPEG olarak kaydeder.Dataset burada büyür.
    # path : dosyanın kaydedileceği yer , frame : kameradan alınan görüntü


# Göz Noktaları Mediapipe Haritası
#P1: sol üst, P2: sağ üst, P3: sağ alt, P4: sol alt, P5: orta üst, P6: orta alt
LEFT_EYE = [33, 160, 158, 133, 153, 144 ]
RIGHT_EYE = [362, 385, 387, 263, 373, 380 ]

#çizim için
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
#thickness=1, circle_radius=1: Çizgiler ve noktalar en ince (1 piksel) olsun ki yüzün detaylarını kapatmasın.
#color=(0, 255, 0): Klasik Matrix yeşili.


# ear için uzaklık hesaplama fonksiyonu
def calculate_distance(p1, p2):
    #iki nokta arasındaki uzaklığı hesaplar
    x_fark = p1.x-p2.x
    y_fark = p1.y-p2.y
    return math.sqrt(x_fark**2 + y_fark**2)


#Göz Açıklık Hesaplama Fonksiyonu
def get_eye_landmarks(face_landmarks, eye_indices):
    # bu fonksiyon mediapipe'ın bulduğu tüm yüz (face_landmarks) alır.
    #bizim istediğimiz eye_indices alır.
    # sadece o koordinatları bir liste yapar.
    
    #boş liste oluşturuyoruz
    eye_points = []
    
    #istediğimiz gözün noktalarını alıyoruz döngü kuruyoruz.
    for idx in eye_indices:
        #face_landmarks.landmark[idx] bize x,y,z koordinatlarını verir.
        point = face_landmarks.landmark[idx]
        eye_points.append(point)
    return eye_points


def calculate_ear_formula(eye_points):
    #Ear hesaplar
    #eye_points listesi içindeki 6 nokta kullanılır.
    # Dikey mesafeler = Göz kapakları arasındaki mesafeler
    dikey1 = calculate_distance(eye_points[1], eye_points[5]) #P2 - P6
    dikey2 = calculate_distance(eye_points[2], eye_points[4]) #P3 - P5
   
    #Yatay mesafe 
    yatay = calculate_distance(eye_points[0], eye_points[3]) #P1 - P4
    if yatay == 0: #sıfıra bölünme hatası
        return 0
    ear = (dikey1 + dikey2) / (2.0 * yatay)
    return ear


#ROI Fonksiyonu = Bölgeyi Belirleme
def extract_eye_roi(frame, eye_landmarks,image_w,image_h,padding=10):
    #frame:tüm kamera görüntüsü eye_landmarka:6 nokta image_w,İmage_h: resim boyutları padding: kenar boşluğu piksel
    #1. adım : koordinatları piksel cinsine çevir
    pixel_points = []
    for lm in eye_landmarks:
        px = int(lm.x * image_w)
        py = int(lm.y * image_h)
        pixel_points.append([px, py])
        
    #2. adım : Nump dizisine çevir : opencv ister.
    pixel_points = np.array(pixel_points)
    
    #3.adım : Kutuyu bul
    x, y, w, h = cv2.boundingRect(pixel_points) #x,y: sol üst köşe w:genişlik h:yükseklik
    
    #4.adım : padding ekle
    x = max(x - padding, 0) # max ile negatif olmasını engelliyoruz
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image_w - x) # min ile taşmasını engelliyoruz
    h = min(h + 2 * padding, image_h - y)
    
    #5.adım : ROI'yi kes
    roi_image = frame[y:y+h, x:x+w] # önce y ekseni (satır) sonra x ekseni (sütun)
    return roi_image


#CNN HAZIRLIK 
def prepare_for_cnn (roi_image,target_size=(64,64)):
    #göz ROI'sini CNN için hazırla
    # ROI boşsa hata vermesin
    if roi_image is None or roi_image.size == 0:
        return None
    
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) #griye çevir
    resized = cv2.resize(gray, target_size) #yeniden boyutlandır
    normalized = resized / 255.0 #normalleştir
    reshaped = np.reshape(normalized, (1, 1, target_size[0], target_size[1]))#şekillendir
    return reshaped

# (GÖZ ROI TEMİZLEME + STANDARTLAŞTIRMA)

def blur_score(gray):
    """
    Bu fonksiyon görüntünün bulanık olup olmadığını anlamak için var.
    Değer küçükse görüntü bulanıktır (net değil demektir).
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def saturated_ratio(gray, low=5, high=250):
    """
    Bu fonksiyon görüntüde aşırı siyah / aşırı beyaz piksellerin oranını bulur.
    Işık patlaması veya çok karanlık görüntüleri elemek için kullanırız.
    """
    total = gray.size
    if total == 0:
        return 1.0
    saturated_pixels = np.sum((gray <= low) | (gray >= high))
    return float(saturated_pixels) / float(total)

def clean_and_normalize_eye_roi(
    roi_bgr,
    target_size=(64, 64),
    min_area=20*20,
    min_lap_var=35.0,
    min_mean=20.0,
    max_mean=235.0,
    max_sat_ratio=0.20,
    use_clahe=True
):
    """
    Dataset'e çöp veri girmesin.

    ROI (göz görüntüsü) şu kontrollerden geçmezse kaydetmiyorum:
    - Çok küçükse
    - Çok karanlık / çok parlaksa
    - Bulanıksa
    - Aşırı siyah/beyaz oranı fazlaysa

    Geçerse:
    - griye çeviriyorum
    - 64x64'e sabitliyorum
    - ışığı biraz dengeliyorum (CLAHE)
    - dataset'e kaydedilecek temiz patch döndürüyorum
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    h, w = roi_bgr.shape[:2]
    if h * w < min_area:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    mean_intensity = np.mean(gray)
    if mean_intensity < min_mean or mean_intensity > max_mean:
        return None

    if blur_score(gray) < min_lap_var:
        return None

    if saturated_ratio(gray) > max_sat_ratio:
        return None

    gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return gray


#SAYAÇLAR
saved_eye_count = 0      # Dataset'e giren temiz göz sayısı
rejected_eye_count = 0   # Kalite filtresinden geçemeyen göz sayısı
total_eye_count = 0      # Kontrol edilen toplam göz sayısı


#modeli başlat
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2, #aynı anda en fazla 2 yüz tespiti
    refine_landmarks=True, #daha fazla yüz detayı için
    #false olursa sadece göz kapağı çevresini verir(468 nokta)
    #true olursa gözün irisini çevreleyen ekstra noktaları da verir(478)
    #kişinin sağa sola baktığını anlamak için buna ihtiyacımız olucak.
    min_detection_confidence=0.5, #algılama
    min_tracking_confidence=0.5 #takip 
)


last_save_time = 0  
#Frame kayıt zamanını kontrol etmek için kullanılan sayaçtır.
#Döngü başlamadan önce 0 atanarak ilk kayıt işlemi başlar.


# Kameradan görüntü al ve yüzleri tespit et
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    #Boyutları al
    image_h, image_w, _ = frame.shape
    
    # BGR'dan RGB'ye çevir
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    #Şuanki zamanı saniye cinsinden verir.
    current_time = time.time()  #Amacımız kayıt aralığını kontrol etmek


    # yüz VARSA (FaceMesh landmark bulduysa)
    if results.multi_face_landmarks:

        # Not: Birden fazla yüz varsa bile label/kayıt için ilk yüzün EAR'ını baz alıyoruz.
        # Bu, datasetin daha tutarlı olmasını sağlar (istersen tüm yüzlerden ortalama da alabiliriz).
        face_landmarks0 = results.multi_face_landmarks[0]

        # Verileri hazırlamak için
        left_points0 = get_eye_landmarks(face_landmarks0, LEFT_EYE)
        right_points0 = get_eye_landmarks(face_landmarks0, RIGHT_EYE)

        # EAR hesapla
        left_ear0 = calculate_ear_formula(left_points0)
        right_ear0 = calculate_ear_formula(right_points0)
        avg_ear = (left_ear0 + right_ear0) / 2.0

        label = label_eye_state(avg_ear)  # etiket üret

        cv2.putText(frame, f"Label: {label.upper()}", #Ekrana etiketi yazdır.
                    (30, 200),  #Yazının ekrandaki konumu
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

        # belirli aralıklarla kaydet
        # last_save_time döngü öncesinde 0 olarak başlatılmalı!
        if current_time - last_save_time >= SAVE_INTERVAL: 
            # open/closed için TAM FRAME yerine GÖZ PATCH kaydediyoruz.
            left_roi0 = extract_eye_roi(frame, left_points0, image_w, image_h)
            right_roi0 = extract_eye_roi(frame, right_points0, image_w, image_h)

            left_patch0 = clean_and_normalize_eye_roi(left_roi0)
            right_patch0 = clean_and_normalize_eye_roi(right_roi0)

            # Sol göz sayacı
            total_eye_count += 1
            if left_patch0 is not None:
                save_frame(left_patch0, label)  #open/closed klasörüne kaydeder.
                saved_eye_count += 1
            else:
                rejected_eye_count += 1

            # Sağ göz sayacı
            total_eye_count += 1
            if right_patch0 is not None:
                save_frame(right_patch0, label) #open/closed klasörüne kaydeder.
                saved_eye_count += 1
            else:
                rejected_eye_count += 1

            last_save_time = current_time #en son kayıt zamanını günceller.


        # Aşağıdaki kısım: çizim + ROI gösterimi + EAR yazıları (tüm yüzler için)
        for face_landmarks in results.multi_face_landmarks:
            # Verileri hazırlamak için
            left_points = get_eye_landmarks(face_landmarks, LEFT_EYE)
            right_points = get_eye_landmarks(face_landmarks, RIGHT_EYE)
            
            # EAR hesapla
            left_ear = calculate_ear_formula(left_points)
            right_ear = calculate_ear_formula(right_points) 
            avg_ear_show = (left_ear + right_ear) / 2.0
            
            # Sol göz
            left_roi = extract_eye_roi(frame, left_points, image_w, image_h)
            if left_roi.size > 0:
                #göze zoom
                left_roi_show = cv2.resize(left_roi, (200, 100))
                cv2.imshow('Sol göz ROI', left_roi_show)

            # Sağ göz
            right_roi = extract_eye_roi(frame, right_points, image_w, image_h)
            if right_roi.size > 0:
                right_roi_show = cv2.resize(right_roi, (200, 100))
                cv2.imshow('Sağ göz ROI', right_roi_show)
                
            #CNN input hazırlama
            left_input = prepare_for_cnn(left_roi)
            right_input = prepare_for_cnn(right_roi)
            # left_input/right_input None olabilir (ROI boşsa). Bu normal.

            #EAR Değerini göster ve yazdır
            cv2.putText(frame, f'Sol EAR: {left_ear:.2f}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    
            cv2.putText(frame, f'Sag EAR: {right_ear:.2f}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Ortalama EAR: {avg_ear_show:.2f}', (30, 150),
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
        #Yüz algılamadıysa ekrana UNFOCUSED yaz.
        cv2.putText(frame, "Label: UNFOCUSED", 
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        #Yüz yokken de dataset toplamak için belirli aralıkla kaydet
        if current_time - last_save_time >= SAVE_INTERVAL:
            save_frame(frame, "unfocused") # unfocused klasörüne kaydeder.
            last_save_time = current_time

    # CANLI SAYAÇ GÖSTERİMİ
    cv2.putText(frame, f"Toplam Goz: {total_eye_count}", (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, f"Kaydedilen: {saved_eye_count}", (30, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Elenen: {rejected_eye_count}", (30, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow('MediaPipe FaceMesh', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
