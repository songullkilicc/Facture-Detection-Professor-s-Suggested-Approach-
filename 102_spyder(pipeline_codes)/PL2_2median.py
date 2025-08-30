import cv2
import numpy as np
import matplotlib.pyplot as plt

# Median Filter uygulama fonksiyonu
def apply_median_filter(image_path, output_path):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Median filter uygulama (3x3 kernel)
    filtered_img = cv2.medianBlur(img, 5)

    # Çıktıyı belirlenen klasöre kaydetme
    cv2.imwrite(output_path, filtered_img)

    # Görüntüyü matplotlib ile gösterme
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('MinMax Normalized Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_img, cmap='gray')
    plt.title('Median Filter Applied')
    plt.axis('off')

    plt.show()

# Görüntü yolu ve çıktı yolu
input_image_path = r"C:\Users\songu\Downloads\outputs\PL2_MinMax_norm.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL2_MinMax_medianfilter.png"

# Median filter uygulama fonksiyonunu çağırma
apply_median_filter(input_image_path, output_image_path)

print(f"Çıktı: {output_image_path} olarak kaydedildi.")