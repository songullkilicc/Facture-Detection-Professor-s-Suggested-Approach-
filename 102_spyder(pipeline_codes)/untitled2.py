import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure, color
from skimage.feature import canny
from skimage.io import imread
from skimage.draw import rectangle
from skimage.color import rgb2gray

# Parametreler
m = 20  # Patch boyutu (m x m)

# Görsel yolları
image_paths = [
    r"C:\Users\songu\Downloads\images\processed_kirik_6.png",
    r"C:\Users\songu\Downloads\images\processed_kirik_3.png",
    r"C:\Users\songu\Downloads\images\processed_kirik_2.png"
]
label_paths = [
    r"C:\Users\songu\Downloads\images\kirik6_label.png",
    r"C:\Users\songu\Downloads\images\kirik3_label.png",
    r"C:\Users\songu\Downloads\images\kirik2_label.png"
]

# Patch alma fonksiyonu
def extract_patch(image, center, m):
    x, y = center
    half_m = m // 2
    patch = image[max(0, y - half_m):y + half_m + 1, max(0, x - half_m):x + half_m + 1]
    return patch

# Görsel üzerinden işlem yapma
for image_path, label_path in zip(image_paths, label_paths):
    # Görseli ve etiketi yükle
    image = imread(image_path)
    label = imread(label_path)

    # RGBA görüntü ise yalnızca RGB'yi al
    if image.shape[2] == 4:  # Eğer 4 kanal (RGBA) ise
        image = image[:, :, :3]  # RGBA'dan RGB'ye geçiş yap

    # Görüntüyü gri tonlamaya dönüştürme
    image_gray = rgb2gray(image)

    # Canny kenar tespiti
    edges = canny(image_gray)

    # Kenarları bul
    contours = measure.find_contours(edges, 0.5)

    # Görseli göster
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    # Kenar noktalarından patch alıp göster
    for contour in contours:
        for point in contour:
            y, x = point  # Kenar noktasının (y, x) koordinatları
            patch = extract_patch(image, (int(x), int(y)), m)

            # Patch'i görselleştir (dikdörtgen)
            ax.add_patch(plt.Rectangle((x - m // 2, y - m // 2), m, m, linewidth=1, edgecolor='red', facecolor='none'))

    # Başlık
    ax.set_title(f"{os.path.basename(image_path)} - Kenar Noktasından Patch Alma")

    # Görseli göster
    plt.show()
