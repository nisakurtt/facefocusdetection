import torch
import torch.nn as nn
import torch.nn.functional as F

#CNN Mimarisi

class EyeStateFocusModel(nn.Module):
    def __init__(self):
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

    def forward(self, x): 
        
        #conv > ReLU > Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        
        #Düzleştirme : Flatten
        x = x.view(-1, 128*6*6)
        
        # Tam bağlantılı katmanlar
        x = F.relu(self.fc1(x)) #fc1 -> ReLU
        x = self.dropout(x)
        
        
        return 