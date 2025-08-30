import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görselin yolu
input_image_path = r"C:\Users\songu\Downloads\kirik_3.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL4_ZScore_Normalized.png"

# Görseli okuma
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Z-Score Normalizasyonu
mean = np.mean(img)
std_dev = np.std(img)
z_score_img = (img - mean) / std_dev

# Z-Score normalizasyonlu görüntüyü 0-255 aralığına sıkıştırma
z_score_img_normalized = cv2.normalize(z_score_img, None, 0, 255, cv2.NORM_MINMAX)

# Z-Score normalizasyonlu görüntüyü kaydetme
cv2.imwrite(output_image_path, z_score_img_normalized)

# Matplotlib ile görüntüyü gösterme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Orijinal Görüntü")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Z-Score Normalizasyonlu Görüntü")
plt.imshow(z_score_img_normalized, cmap='gray')
plt.axis('off')

plt.show()

# Çıktı bilgisi
print(f"Görüntü Z-Score Normalizasyonu uygulandı ve kaydedildi: {output_image_path}")