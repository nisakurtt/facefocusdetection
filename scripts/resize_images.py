import cv2
import os


SOURCE_DIR = "/Users/nisakurt/Desktop/proje/prepared/face/"     
TARGET_DIR = "/Users/nisakurt/Desktop/proje/final_dataset"     
IMAGE_SIZE = (224, 224)
a = 2
b = 3
c = "ben nisa" 
d = "bende nisa memnun oldum"
f = "Bende nehir gençler" 
e="hosgeldin cnm"
h="hb"

def create_target_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["focused", "not_focused", "sleepy"]:
            os.makedirs(f"{TARGET_DIR}/{split}/{cls}", exist_ok=True)

def get_all_files():
    data = {}
    for cls in ["focused", "not_focused", "sleepy"]:
        folder = f"{SOURCE_DIR}/{cls}"
        files = [f"{folder}/{f}" for f in os.listdir(folder) if f.lower().endswith((".jpg",".jpeg",".png"))]
        data[cls] = files
    return data

def split(files):
    import random
    random.shuffle(files)
    n = len(files)
    train = int(n * 0.8)
    val = int(n * 0.1)
    return files[:train], files[train:train+val], files[train+val:]

def process_files(files, split, cls):
    for file_path in files:
        img = cv2.imread(file_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)

        file_name = os.path.basename(file_path)
        save_path = f"{TARGET_DIR}/{split}/{cls}/{file_name}"

        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def main():
    create_target_dirs()
    data = get_all_files()

    for cls, files in data.items():
        train_files, val_files, test_files = split(files)

        process_files(train_files, "train", cls)
        process_files(val_files, "val", cls)
        process_files(test_files, "test", cls)

    print("✔ Tüm fotoğraflar 224x224'e dönüştürüldü ve final dataset hazır!")

if __name__ == "__main__":
    main()
