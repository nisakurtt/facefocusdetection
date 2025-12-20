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


class CameraPipeline:
    def __init__(self,camera_id: int = 0,target_size: Optional[Tuple[int, int]] = (640,480)):
        self.camera_id = camera_id #camera_id : 0 dahili kameradır.
        self.target_size = target_size #pipeline boyunca frame boyutunu standart tuttar.
        
        self.cap = cv2.VideoCapture(self.camera_id) #Kamerayı başlatır.Donanımsal.
        if not self.cap.isOpened(): #Kamera açılmazsa hata verir.
            raise RuntimeError(f"Kamera açılamadı. camera_id = {self.camera_id}")
        
        self.prev_time = time.time() #fps için bir önceki frame'in zamanı
        self.frame_id = 0 #frame sayacı debug + takip
        self.fps = 0.0 #anlık fps değeri    
        
        #bir adet frame + meta döndürür.
    def read(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
             # ret : frame alındı mı, frame : alınan görüntü (numpy array, bgr formatında mı?)
            ret, frame = self.cap.read() #kameradan bir anlık fotoğraf çeker. 'ret' başarı durumudur
            if not ret or frame is None:  #kamera anlık olarak frame vermezse sistemi çökertme.
                return None, None
            
            
            
            #Tam frame resize edilir. ROI değil. analiz yok. sadece performans ve tuttarlılık.
            if self.target_size is not None:
                frame = cv2.resize(frame, self.target_size) 
                
            #fps hesaplama
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now
            if dt > 0:
                self.fps = 1.0 / dt #fps = 1/-iki frame arası süre-

            self.frame_id += 1
            
            meta = {
                "frame_id": self.frame_id,
                "timestamp": now,
                "fps": round(self.fps, 2),
                "shape": frame.shape #(height, width, channels)
            }
            return frame, meta

    def release(self) -> None:
            if self.cap is not None:
                self.cap.release() #kamera kaynağını serbest bırakır.


class EyeStatePredictor:
  def __init__(
    self,
    model_path="eye_state_cnn.pth",   # Eğitilmiş modelin dosya yolu
    device=None,
    class_map=None,
    ear_threshold=0.23,   # EAR için açık/kapalı eşiği [cite: 79]
    cnn_open_threshold=0.50,    # CNN olasılık eşiği
    cnn_conf_threshold=0.60, # CNN'e güvenme eşiği (%60 altıysa EAR'a sor)
    norm_mean=None,
    norm_std=None,
  ):
        
        # İşlemciyi (GPU veya CPU) belirliyoruz.
    self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

    # 2 sınıflı model
    self.model = EyeStateFocusModel(num_classes=2).to(self.device)
    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    self.model.eval()  # Modeli eğitim modundan test (tahmin) moduna alıyoruz.

    # 2-sınıf mapping: BUNU 3-sınıflı class_to_idx ile karıştırma.
    # Training'de open/closed hangi sıradaysa onu yaz.
    self.class_map = class_map if class_map else {0: "closed", 1: "open"}

    self.ear_threshold = ear_threshold
    self.cnn_open_threshold = cnn_open_threshold
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
        # 'open' sınıfının olasılığını ayrıca sakla.
    prob_open = float(probs[1].item()) if 1 in self.class_map and self.class_map[1] == "open" else None
    return {"label": label, "conf": conf, "prob_open": prob_open}

  def fuse(self, ear_val, cnn_out):  # EAR ve CNN sonuçlarını birleştirerek son kararı verir.
        # Senaryo 1: ROI alınamadıysa veya CNN çalışmadıysa sadece EAR'a güven.
    # CNN yoksa -> EAR
    if cnn_out is None:
      return "open" if ear_val >= self.ear_threshold else "closed"

    # Senaryo 2: CNN tahmini yaptı ama güven skoru düşükse (Örn: %55), EAR'a sor.
        # CNN güveni düşükse -> EAR
    if cnn_out["conf"] < self.cnn_conf_threshold:
      return "open" if ear_val >= self.ear_threshold else "closed"

        # Senaryo 3: CNN kendine güveniyorsa (%60+), CNN'in dediğini kabul et.
    # CNN güvenliyse -> CNN
    return cnn_out["label"]

  def get_prediction(self, frame, eye_landmarks, image_w, image_h):
        # 1. Matematiksel EAR hesapla. [cite: 91]
    ear_val = calculate_ear_formula(eye_landmarks)
        # 2. Göz bölgesini (ROI) kesip al. [cite: 94]
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



def main():
    cam = CameraPipeline(camera_id=0,target_size=(640,480))
    
    try:
        while True: # kamera bağlantısı koparsa kontrollü çıkış
            frame, meta = cam.read()
            if frame is None:
                print("Kamera görüntüsü alınamadı.")
                break
            
            #DEBUG : fps göster - karar yok-
            cv2.putText(
                frame, 
                f"FPS: {meta['fps']} | Frame: {meta['frame_id']}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )
            cv2.imshow("CameraPipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): #q ile çıkış
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__": #pipeline çalışıyor mu, fps akıyor mu, kamera stabil mi?
    main()
   
#Neden iki sınıf : camerapipeline = öğretici /debug/basir test
#theadedcamerapipeline = gerçek uygulama için

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
    
    try:
        while True:
            frame, meta = cam.read()
            if frame is None:
                continue #frame yüklenmesini bekle
            
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
            cv2.imshow("ThreadedCameraPipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main() 