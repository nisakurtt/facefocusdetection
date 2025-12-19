import cv2 
import time 
from typing import Optional, Tuple,Dict, Any # Kodun okunabilirliği ve hata payını düşürmek için tip belirleyiciler.
import threading # arka planda pararlel işlem yapmak için.

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