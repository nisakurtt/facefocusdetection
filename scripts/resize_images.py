import os
import cv2
import time

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
    timestamp = time.strftime("%Y%m%d_%H%M%S_%f")  #Anlık tarih + saat + milisaniye
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


#Şuanki zamanı saniye cinsinden verir.
current_time = time.time()  #Amacımız kayıt aralığını kontrol etmek

# yüz VARSA (FaceMesh landmark bulduysa)
if results.multi_face_landmarks:
    label = label_eye_state(avg_ear)  # etiket üret

    cv2.putText(frame, f"Label: {label.upper()}", #Ekrana etiketi yazdır.
                (30, 200),  #Yazının ekrandaki konumu
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    # belirli aralıklarla kaydet
    # last_save_time döngü öncesinde 0 olarak başlatılmalı!
    if current_time - last_save_time >= SAVE_INTERVAL: 
        save_frame(frame, label) #open/closed klasörüne kaydeder.
        last_save_time = current_time #en son kayıt zamanını günceller.

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
        
        
        
        
  # Döngü öncesine bunu ekle (çok önemli)

last_save_time = 0  
#Frame kayıt zamanını kontrol etmek için kullanılan sayaçtır.
#Döngü başlamadan önce 0 atanarak ilk kayıt işlemi başlar.




