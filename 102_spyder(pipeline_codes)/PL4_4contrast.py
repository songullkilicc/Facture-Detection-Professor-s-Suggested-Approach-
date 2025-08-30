import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görselin yolu
input_image_path = r"C:\Users\songu\Downloads\outputs\PL4_Sobel_Edge_Detected.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL4_Histogram_Equalized.png"

# Görseli okuma
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Histogram Equalization uygulama
hist_eq = cv2.equalizeHist(img)

# Histogram eşikleme yapılmış görüntüyü kaydetme
cv2.imwrite(output_image_path, hist_eq)

# Matplotlib ile görüntüyü gösterme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Sobel Edge Detection Uygulandı")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Histogram Equalization Uygulandı")
plt.imshow(hist_eq, cmap='gray')
plt.axis('off')

plt.show()

# Çıktı bilgisi
print(f"Görüntü Histogram Equalization uygulandı ve kaydedildi: {output_image_path}")
