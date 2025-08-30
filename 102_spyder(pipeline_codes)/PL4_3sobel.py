import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görselin yolu
input_image_path = r"C:\Users\songu\Downloads\outputs\PL4_ZScore_Median_Filtered.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL4_Sobel_Edge_Detected.png"

# Görseli okuma
img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

# Sobel Edge Detection uygulama
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # X yönünde Sobel
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Y yönünde Sobel
sobel_edge = cv2.magnitude(sobel_x, sobel_y)  # Sobel kenarlarını birleştirme

# Sobel kenar tespiti yapılmış görüntüyü kaydetme
sobel_edge = np.uint8(np.absolute(sobel_edge))  # Değerleri uint8'e dönüştürme
cv2.imwrite(output_image_path, sobel_edge)

# Matplotlib ile görüntüyü gösterme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Median Filter Uygulandı")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sobel Edge Detection Uygulandı")
plt.imshow(sobel_edge, cmap='gray')
plt.axis('off')

plt.show()

# Çıktı bilgisi
print(f"Görüntü Sobel Edge Detection uygulandı ve kaydedildi: {output_image_path}")

