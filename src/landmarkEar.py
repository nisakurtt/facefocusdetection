import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils 
mp_face_mesh = mp.solutions.face_mesh #yüz modülü

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
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY) #griye çevir
    resized = cv2.resize(gray, target_size) #yeniden boyutlandır
    normalized = resized / 255.0 #normalleştir
    reshaped = np.reshape(normalized, (1, 1, target_size[0], target_size[1]))#şekillendir
    return reshaped


#modeli başlat
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2, #aynı anda en fazla 2 yüz tespiti
    refine_landmarks=True, #daha fazla yüz detayı için
    #false olursa sadece göz apağı çevresini verir(468 nokta)
    #true olursa gözün irisini çevreleyen ekstra noktaları da verir(478)
    #kişinin sağa sola baktığını anlamak için buna ihtiyacımız olucak.
    min_detection_confidence=0.5, #algılama
    min_tracking_confidence=0.5 #takip 
)

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

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Verileri hazırlamak için
            left_points = get_eye_landmarks(face_landmarks, LEFT_EYE)
            right_points = get_eye_landmarks(face_landmarks, RIGHT_EYE)
            
            # EAR hesapla
            left_ear = calculate_ear_formula(left_points)
            right_ear = calculate_ear_formula(right_points) 
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Sol göz
            left_roi = extract_eye_roi(frame, left_points, image_w, image_h)
            if left_roi.size > 0:
                #göze zoom
                left_roi = cv2.resize(left_roi, (200, 100))
                cv2.imshow('Sol göz ROI', left_roi)
            # Sağ göz
            right_roi = extract_eye_roi(frame, right_points, image_w, image_h)
            if right_roi.size > 0:
                right_roi = cv2.resize(right_roi, (200, 100))
                cv2.imshow('Sağ göz ROI', right_roi)
                
            #CNN input hazırlama
            left_input = prepare_for_cnn(left_roi)
            right_input = prepare_for_cnn(right_roi)
            
            #EAR Değerini göster ve yazdır
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

    cv2.imshow('MediaPipe FaceMesh', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()