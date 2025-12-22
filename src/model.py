import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset 
from torchvision import transforms, datasets


BASE_DIR = "dataset" # Veri setinin ana klasörü 
BATCH_SIZE = 32 # Modeli eğitirken aynı anda işlenecek örnek sayısı
TARGET_SIZE = (64, 64) # Tüm görselleri standartlaştırmak için 64×64
TRAIN_RATIO = 0.8 # Verinin %80’i eğitim, %20’si doğrulama.

# Öğrenme hızı:
# Modelin ağırlıkları ne kadar hızlı değiştireceğini belirler.
# Çok büyük olursa model kararsız olur, çok küçük olursa öğrenme çok yavaşlar.
LR = 1e-3

# Eğer bilgisayarda ekran kartı (GPU) varsa onu kullan, yoksa CPU kullan.
# GPU kullanmak eğitimi ciddi şekilde hızlandırır.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Eğitilmiş modelin kaydedileceği dosya adı.
# Bu dosya daha sonra gerçek zamanlı sistemde kullanılacak.
SAVE_PATH = "eye_state_cnn.pth"

# TRANSFORMLAR (DÖNÜŞÜMLER)
# Eğitim setindeki görüntülere uygulanacak dönüşümler (Augmentation içerir)
train_transforms = transforms.Compose([
    transforms.ToTensor(),  # OpenCv'den gelen NumPy görüntüyü Tensor'a çeviri.Değerleri [0, 255]den [0, 1] aralığına getirir.Gri görüntü ise (1,H,W) olur.
    transforms.RandomHorizontalFlip(p=0.5), # %50 ihtimalle görüntüyü sağ-sol çevirir.Modelin ters yönde duran gözleri de öğrenmesini kolaylaştırır.
    transforms.RandomRotation(10), # -10 ile +10 derece arasında döndürür.Kamera açısı değişince model daha dayanıklı olur.
    # Rastgele Gürültü/Parlaklık Değişimi: Görüntüye rastgele değerler ekle, değerlerin [0.0, 1.0] aralığında kalmasını sağla.
    transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.1, 0.0, 1.0))
])

# Doğrulama setindeki görüntülere uygulanacak dönüşümler (Sadece formatlama, Augmentation YOK)
val_transforms = transforms.Compose([
    transforms.ToTensor() # Doğrulamada veriyi bozmak istemediğin için sadece Tensor'a dönüştür ve normalize et.
])

# IMAGEFOLDER İÇİN GRİ LOADER (OpenCV)
# ImageFolder'ın varsayılan PIL(RGB) tabanlı yükleyicisi yerine, OpenCV kullanarak gri tonlamalı okuma ve yeniden boyutlandırma yapar.
def gray_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # Gri tonlamalı (tek kanal) olarak oku
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, TARGET_SIZE) # Görüntüyü 64x64'e yeniden boyutlandır
    return img  # Sonuç NumPy dizisi (H,W) olarak döner, ToTensor() bunu (1,H,W) yapar.

# ANA DATALOADER OLUŞTURMA FONKSİYONU
def create_dataloaders_simplified(base_dir=BASE_DIR, batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO):
    print("\n--- DataLoader'lar hazırlanıyor (düzeltilmiş) ---")

    # 1. ImageFolder nesnelerinin oluşturulması:
    # Aynı kök dizinden iki ayrı Dataset oluşturulur.
    # ds_train: Augmentation dönüşümlerini kullanır (train_transforms)
    ds_train = datasets.ImageFolder(root=base_dir, transform=train_transforms, loader=gray_loader)
    # ds_val: Sadece formatlama dönüşümlerini kullanır.Augmentationsız (val_transforms)
    #Bu sayede val setine augmentation sızmıyor.
    ds_val   = datasets.ImageFolder(root=base_dir, transform=val_transforms, loader=gray_loader)

    total = len(ds_train) # Veri setindeki toplam örnek sayısı(%80)
    train_count = int(total * train_ratio) # Eğitim örnek sayısı
    val_count = total - train_count # Doğrulama örnek sayısı(%20)

    # 2. İndekslerin Rastgele Bölünmesi (Split):
    # random_split, tüm veri setini değil, sadece indeks aralığını rastgele böler.
    # Böylece train ve val aynı örnekleri paylaşmıyor.
    idx_train, idx_val = random_split(range(total), [train_count, val_count])

    # 3. Alt Kümelerin (Subset) Oluşturulması:
    # Bölünen indeksler kullanılarak, doğru transformlara sahip dataset'lerden alt kümeler üretilir.
    # Bu, Eğitim ve Doğrulama setlerinin *farklı* dönüşümlere sahip olmasını sağlar.
    # Train subset,train transform'lu dataset'ten gelir.
    # Val subset,val transform'lu dataset'ten gelir.
    train_subset = Subset(ds_train, idx_train.indices) # Eğitim seti (Augmentation alır)
    val_subset   = Subset(ds_val,   idx_val.indices)   # Doğrulama seti (Augmentation almaz)

    # 4. DataLoader'ların Oluşturulması (Batch Yönetimi):
    # DataLoader, alt kümeleri alıp onları gruplar (Batch) halinde yüklemeye yarar.
    # Eğitimde karıştırmak iyidir genelleme artar.
    # Validation'da sabit sıra sorun değil.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True) # Eğitimde karıştır (shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False) # Doğrulamada karıştırma (shuffle=False)
   
    
    # Konsol Bilgilendirmesi
    print("class_to_idx:", ds_train.class_to_idx) # Veri setinin etiket-indeks eşleşmesini gösterir (Örnek: {'closed': 0, 'open': 1})
    print(f"Toplam: {total} | Train: {len(train_subset)} | Val: {len(val_subset)}")
    print(f"Train batch: {len(train_loader)} | Val batch: {len(val_loader)}")

    return train_loader, val_loader, ds_train.class_to_idx




#CNN Mimarisi

class EyeStateFocusModel(nn.Module):
    def __init__(self, num_classes=3):
        super(EyeStateFocusModel, self).__init__()
        #Katmanları Tanımlama
        
        #1. Konvolüsyonel Katmanlar > gri tonlama
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
                            #girdi tek kanallı gri tonlama,32 filtre,filtrenin boyutu 3x3 piksel.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                            #filtrelerden sonra boyutları küçültmek için kullanılır.
        # girdi(1,64,64) -> conv1 -> (32,62,62) -> pool -> (32,31,31)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # girdi(32,31,31) -> conv2 -> (64,29,29) -> pool -> (64,14,14)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        # girdi(64,14,14) -> conv3 -> (128,12,12) -> pool -> (128,6,6)
       
       
       #katmanlar arası bağlantı
        self.fc1 = nn.Linear(128*6*6, 128) # tek boyutlu vektöre dönüştürme
        self.dropout = nn.Dropout(p=0.5) 

        # Bu katman modelin EN SON KARAR VERDİĞİ yerdir.
        # Burada artık "bu göz OPEN mi CLOSED mi?" sorusuna cevap verilir.
        # num_classes = 3 ise çıktı 3 sayı olur.
        # Bu sayılar olasılık değil, skor (logits) değerleridir.
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # CNN katmanlarından çıkan çok boyutlu veriyi tek boyutlu bir vektöre çeviriyoruz. Çünkü fully connected katmanlar düz veri ister.
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Burada model sınıflandırma skorlarını üretir. Örneğin: [2.3, -1.1] gibi. CrossEntropyLoss bu skorları kullanır.
        x = self.fc2(x)
        # Eğer bu return olmazsa model hiçbir şey üretmez
        # ve eğitim tamamen imkansız hale gelir.
        return x
    


def run_epoch(model, loader, criterion, optimizer=None):
    """
    Bu fonksiyon modelin 1 epoch boyunca çalışmasını sağlar.

    Eğer optimizer verilirse:
        → TRAIN yapılır (model öğrenir)

    Eğer optimizer None ise:
        → VALIDATION yapılır (model öğrenmez, sadece test edilir)
    """

    # TRAIN mi VALIDATION mı?
    is_train = optimizer is not None

    # Train modunda dropout vs aktif olur
    # Val modunda kapatılır
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0      # Toplam hata
    correct = 0           # Doğru tahmin sayısı
    total = 0             # Toplam örnek sayısı

    # Validation sırasında gereksiz gradient hesaplamasını kapat
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:

            # Görüntüleri ve etiketleri GPU/CPU'ya taşı
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Model tahmin yapıyor
            outputs = model(images)

            # Tahmin ile gerçek arasındaki fark (loss)
            loss = criterion(outputs, labels)

            # Eğer TRAIN ise:
            if is_train:
                optimizer.zero_grad()   # Önceki adımın etkisini sil
                loss.backward()         # Hatanın geriye yayılması
                optimizer.step()        # Ağırlıkları güncelle

            # İstatistikleri tut
            total_loss += loss.item() * images.size(0)

            # En yüksek skoru alan sınıf = modelin tahmini
            preds = torch.argmax(outputs, dim=1)

            # Kaç tanesi doğru?
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Ortalama loss ve accuracy
    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
    

if __name__ == "__main__":
    # Kod dosyası doğrudan çalıştırıldığında bu kısım çalışır ve DataLoader'ı test eder.
    train_loader, val_loader, class_to_idx = create_dataloaders_simplified()
    # İlk batch'i alıp boyutlarını kontrol et
    images, labels = next(iter(train_loader))
    # Batch boyutu: (Batch Sayısı, Kanal Sayısı, Yükseklik, Genişlik) -> (B, 1, 64, 64)
    print("Batch images:", images.shape)
    print("Batch labels:", labels.shape)

    # MODEL EĞİTİMİ (TRAIN)
    # Kaç sınıf var? (open_eye / closed_eye → 2)
    # sınıf sayısını otomatik al
    num_classes = len(class_to_idx)


    # Modeli oluştur
    model = EyeStateFocusModel(num_classes=num_classes).to(DEVICE)
    print("Eğitilecek sınıf sayısı:", num_classes, class_to_idx)
    # Kayıp fonksiyonu: sınıflandırma için standart
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    EPOCHS = 10

    # Eğitim döngüsü
    for epoch in range(EPOCHS):


        # TRAIN

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # VALIDATION
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion
        )

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f}")

    # Eğitilmiş modeli kaydet
    torch.save(model.state_dict(), "eye_state_cnn.pth")
    print("Model kaydedildi: eye_state_cnn.pth")