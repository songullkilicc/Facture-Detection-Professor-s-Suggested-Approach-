import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görselin yolu
input_image_path = r"C:\Users\songu\Downloads\outputs\PL4_ZScore_Normalized.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL4_ZScore_Median_Filtered.png"

# Görseli okuma
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Median Filter uygulama
median_filtered_img = cv2.medianBlur(img, 5)  # 5x5 pencereli median filter

# Median filter uygulanmış görüntüyü kaydetme
cv2.imwrite(output_image_path, median_filtered_img)

# Matplotlib ile görüntüyü gösterme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Z-Score Normalizasyonlu Görüntü")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Median Filter Uygulandı")
plt.imshow(median_filtered_img, cmap='gray')
plt.axis('off')

plt.show()

# Çıktı bilgisi
print(f"Görüntü Median Filter uygulandı ve kaydedildi: {output_image_path}")