import cv2
import numpy as np
import matplotlib.pyplot as plt

# MinMax Normalizasyonu fonksiyonu
def minmax_norm(image_path, output_path):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Min-Max normalizasyonu (0-255 aralığına dönüştürme)
    min_val = np.min(img)
    max_val = np.max(img)
    norm_img = ((img - min_val) / (max_val - min_val)) * 255

    # Çıktıyı belirlenen klasöre kaydetme
    cv2.imwrite(output_path, norm_img)

    # Görüntüyü matplotlib ile gösterme
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(norm_img, cmap='gray')
    plt.title('MinMax Normalized')
    plt.axis('off')

    plt.show()

# Görüntü yolu ve çıktı yolu
input_image_path = r"C:\Users\songu\Downloads\kirik_3.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL2_MinMax_norm.png"

# MinMax normalizasyonunu çağırma
minmax_norm(input_image_path, output_image_path)

print(f"Çıktı: {output_image_path} olarak kaydedildi.")
