import cv2 
import time 
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple,Dict, Any # Kodun okunabilirliği ve hata payını düşürmek için tip belirleyiciler.
import threading # arka planda pararlel işlem yapmak için.
from model import EyeStateFocusModel  # Kişi 1'in hazırladığı CNN model yapısını içeri aktarıyoruz.
from ear_utils import calculate_ear_formula
from roi_utils import extract_eye_roi

# MediaPipe canlı tahmin için gerekli
import mediapipe as mp

# Aşama 1'de kullanılan göz landmark indexleri
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def get_eye_landmarks(face_landmarks, eye_indices):
    """
    Aşama 1'de kullanılan fonksiyonun aynısı.
    Yüz landmarklarından sadece göz noktalarını alır.
    """
    eye_points = []
    for idx in eye_indices:
        eye_points.append(face_landmarks.landmark[idx])
    return eye_points


class EyeStatePredictor:
  def __init__(
    self,
    model_path="eye_state_cnn.pth",   # Eğitilmiş modelin dosya yolu
    device=None,
    class_map=None,
    ear_threshold=0.23,   # EAR için açık/kapalı eşiği [cite: 79]
    cnn_conf_threshold=0.60, # CNN'e güvenme eşiği (%60 altıysa EAR'a sor)
    norm_mean=None,
    norm_std=None,
  ):
        
        # İşlemciyi (GPU veya CPU) belirliyoruz.
    self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    # 3 sınıflı model (closed_eye / open_eye / unfocused)
    self.model = EyeStateFocusModel(num_classes=3).to(self.device)
    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    self.model.eval()  # Modeli eğitim modundan test (tahmin) moduna alıyoruz.

    # 3-sınıf mapping: training'deki class_to_idx ile UYUMLU olmalı
    # (Sende eğitim çıktısı: {'closed_eye': 0, 'open_eye': 1, 'unfocused': 2})
    self.class_map = class_map if class_map else {0: "closed_eye", 1: "open_eye", 2: "unfocused"}

   

    self.ear_threshold = ear_threshold
    self.cnn_conf_threshold = cnn_conf_threshold

    # Eğitimde Normalize kullanıldıysa burada da aynı olmalı
    self.norm_mean = norm_mean
    self.norm_std = norm_std

  def preprocess_for_cnn(self, roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
      return None
    
        # Görüntüyü gri tonlamaya çevirip 64x64 boyutuna getiriyoruz (Eğitimdeki gibi). [cite: 94, 132]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        # NumPy dizisini PyTorch Tensor'una çevirip [0, 1] arasına normalize ediyoruz. [cite: 130]
    x = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0

    
        # Eğer eğitimde özel bir ortalama/standart sapma kullanıldıysa uyguluyoruz.
    if self.norm_mean is not None and self.norm_std is not None:
      mean = torch.tensor(self.norm_mean).view(1, 1, 1, 1)
      std = torch.tensor(self.norm_std).view(1, 1, 1, 1)
      x = (x - mean) / std

    return x.to(self.device)

  def predict_cnn(self, roi_bgr):  #CNN modelini kullanarak gözün durumunu tahmin eder.
    x = self.preprocess_for_cnn(roi_bgr)
    if x is None:
      return None # CNN yok

    with torch.no_grad():
      logits = self.model(x)       # Modelden ham skorları (logits) al. [cite: 150]
      probs = F.softmax(logits, dim=1)[0]  # Skorları olasılığa çevir (Örn: %80 açık, %20 kapalı).
      pred_idx = int(torch.argmax(probs).item())  # En yüksek olasılıklı sınıfın indeksini al. [cite: 156]
      conf = float(probs[pred_idx].item())    # Bu tahminin güven skorunu al (% kaç emin?).

    label = self.class_map[pred_idx]
    return {
    "label": label,
    "conf": conf
}

  def fuse(self, ear_val, cnn_out):
    # CNN yoksa -> EAR ile sadece open_eye / closed_eye kararı verilir
    if cnn_out is None:
        return "open_eye" if ear_val >= self.ear_threshold else "closed_eye"

 #NOT: Asıl "unfocused" kararı MediaPipe ile (yüz yoksa) veriliyor. Ama model eğitiminde unfocused sınıfı da olduğu için,CNN bazen düşük kalite / yanlış ROI durumlarını "unfocused" diye işaretleyebilir.
 # Bu yüzden burada ekstra bir güvenlik kuralı olarak bırakıldı.

    if cnn_out["label"] == "unfocused" and cnn_out["conf"] >= self.cnn_conf_threshold:
        return "unfocused"

    # CNN güveni düşükse -> EAR'a dön (open/closed)
    if cnn_out["conf"] < self.cnn_conf_threshold:
        return "open_eye" if ear_val >= self.ear_threshold else "closed_eye"

    # CNN güvenliyse -> CNN sonucunu kabul et
    return cnn_out["label"]


  def get_prediction(self, frame, eye_landmarks, image_w, image_h):
        # 1. Matematiksel EAR hesapla. [deneysel]
    ear_val = calculate_ear_formula(eye_landmarks)
        # 2. Göz bölgesini (ROI) kesip al. [deneysel]
    roi = extract_eye_roi(frame, eye_landmarks, image_w, image_h)
        # 3. CNN tahminini al.
    cnn_out = self.predict_cnn(roi)
        # 4. İki sonucu birleştirip (Fusion) son durumu belirle.
    final_status = self.fuse(ear_val, cnn_out)
        
        # Sonuçları bir paket (dictionary) halinde döndür.
    return {
      "ear_value": float(round(ear_val, 3)),
      "cnn_label": None if cnn_out is None else cnn_out["label"],
      "cnn_conf": None if cnn_out is None else float(round(cnn_out["conf"], 3)),
      "final_status": final_status,
      "roi_ok": roi is not None and roi.size != 0,
    }



# NOT: Bu dosyada sınıflandırma 3 sınıf:
# closed_eye / open_eye / unfocused
# (unfocused: yüz yok ya da model düşük güven / kötü ROI senaryoları)
#
# ThreadedCameraPipeline: gerçek zamanlı akışta takılmayı azaltmak için kullanılır.


class ThreadedCameraPipeline:
    #amacı : kamerayı ayrı bir thread ' de sürekli okutmak,
    #.     : ana thread'e en güzel frame'i vermek.  -> bu takılmasız görüntü demektir.
    
    
    #thread hazırlığı
    def __init__(self,camera_id: int =0 ,target_size: Optional[Tuple[int, int]] = (640,480)):
        self.camera_id = camera_id
        self.target_size = target_size
        
        #Kamerayı aç
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı. camera_id = {self.camera_id}")
        
        #thread kontrolü, çalışsın mı?
        self.running = True
        
        #paylaşılan veri (kamera thread -> ana thread)
        self.frame = None
        self.meta = None
        
        #Fps hesaplama değişkenleri
        self.prev_time = time.time()
        self.frame_id = 0
        self.fps = 0.0
        
        #thread kilidi
        self.lock = threading.Lock()
        
        #Kamera okuma thread'i başlat
        self.thread = threading.Thread(target=self.update,daemon=True)
        # daemon = True program kapandığında thread'in de kapanmasını sağlar
        self.thread.start()
    def update(self):
        #sürekli kameradan frame okuma
        while self.running:
            ret,frame = self.cap.read() 
            if not ret:
                continue
            
            if self.target_size is not None:
                frame = cv2.resize(frame,self.target_size)
            
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            if dt > 0:
                self.fps = 1.0 / dt
            
            self.frame_id += 1
            
            meta = {
                "frame_id": self.frame_id,
                "timestamp": now,
                "fps": round(self.fps,2),
                "shape": frame.shape
            }   
            with self.lock: #paylaşılan verilere erişim için kilit
                self.frame = frame #veriyi güncelle,güvenli yere koy.
                self.meta = meta   #metayı güncelle  
                
    def read(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
            with self.lock: #paylaşılan verilere erişim için kilit
               if self.frame is None:
                   return None, None
               #veri güvenliği için kopya döndür
               return self.frame.copy(), self.meta.copy()
    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join() #thread'in bitmesini bekle
        if self.cap is not None:
            self.cap.release() #kamera kaynağını serbest bırak
            
#Çalıştırma Kısmı 
def main():
    #performans testi için ThreadedCameraPipeline kullan
    cam = ThreadedCameraPipeline(camera_id=0,target_size=(640,480))

    # 2. Tahmin Modülünü Başlat (Kişi 2)
    predictor = EyeStatePredictor(model_path="eye_state_cnn.pth")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

    
    try:
        while True:
            frame, meta = cam.read()
            if frame is None:
                continue #frame yüklenmesini bekle


            # Varsayılan: yüz yoksa unfocused
            final_status = "unfocused"
            left_status = None
            right_status = None

            image_h, image_w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face0 = results.multi_face_landmarks[0]

                left_pts = get_eye_landmarks(face0, LEFT_EYE)
                right_pts = get_eye_landmarks(face0, RIGHT_EYE)

                left_out = predictor.get_prediction(frame, left_pts, image_w, image_h)
                right_out = predictor.get_prediction(frame, right_pts, image_w, image_h)

                left_status = left_out["final_status"]
                right_status = right_out["final_status"]

                # İki göz birleşimi (öncelik: unfocused > closed_eye > open_eye)
                if left_status == "unfocused" or right_status == "unfocused":
                    final_status = "unfocused"
                elif left_status == "closed_eye" or right_status == "closed_eye":
                    final_status = "closed_eye"
                else:
                    final_status = "open_eye"


            #FPS ve Bilgileri Ekrana yazdır
            cv2.putText(
                frame, 
                f"FPS: {meta['fps']} | Frame: {meta['frame_id']}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )
            cv2.putText(
                frame,
                f"STATUS: {final_status.upper()}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.imshow("ThreadedCameraPipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main() 