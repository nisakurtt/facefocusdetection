# Face Focus Detection
# Nisa Nur Kurt / Ayşe Nisa Yüksel / Heja Nehir Ecer
#veri seti büyüklüğünden dolayı yüklenmiyor. Hata alıyoruz en kısa sürede yükleyeceğiz.

#Görev Dağılımı

Nisa Nur Kurt
Görev/ Odak	                               Açıklama	                                                  Çıktı
Ortam Kurulumu & Araçlar	          Python, OpenCV, Mediapipe, NumPy kurulumu, sanal ortam.	     env_setup.py, README.md
EAR Fonksiyonu Geliştirme          	Mediapipe göz landmark’larından EAR hesaplayan fonksiyon.	   ear_utils.py
EAR Doğruluk Testi	                Hazır göz-kırpma videolarıyla doğruluk & eşik testi.	       ear_test_results.csv
Görselleştirme	                    EAR zaman grafikleri, threshold karşılaştırma grafikleri.	   focus_graphs.png
GUI’ye EAR Entegrasyonu	            Kamera akışı + EAR değerinin canlı GUI’de gösterimi.	—


Ayşe Nisa Yüksel
Görev / Odak	                             Açıklama	                                                   Çıktı
Veri ve Dosya Yapısı İncelemesi    	EAR verileri okunur, sütun yapısı tanımlanır.	                data_summary.txt
EAR Threshold Belirleme	            Ortalama, medyan, varyans ile “odak ↓” eşiği belirlenir.	    ear_threshold.txt
İstatistik Analizi	                Kullanıcılara göre ortalama odak seviyeleri analizi.	        ear_stats.csv
GUI & Canlı Çizim	                  Canlı olarak göz bölgesi & EAR değeri GUI’de overlay yapılır.	gui_face_demo.py
Performans & FPS Analizi	          Sistem FPS & gecikme testleri, performans raporu.	—


Heja Nehir Ecer
Görev / Odak	                             Açıklama	                                                    Çıktı
Kamera Testi	                       Webcam açılır, FPS ölçülür, stabil çalışma doğrulanır.	       camera_test.py
Focus Score Hesaplama	               EAR + zaman bazlı düşüş → Focus Score formülü geliştirilir.	 focus_logic.py
Focus Süresi Analizi	               Kullanıcının ne kadar süre odakta kaldığı raporlanır.	       focus_summary.csv
Uyarı Sistemi & Gösterge	           Düşük focus durumunda sesli/renkli uyarı sistemi.	           focus_display.py
Demo + Sunum + Raporlama	           1 dakikalık demo videosu + final rapor + slayt.	—
