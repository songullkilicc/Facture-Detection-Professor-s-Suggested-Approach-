import cv2
import matplotlib.pyplot as plt

# Görüntülerin yolları
original_image_path = r"C:\Users\songu\Downloads\kirik_3.png"
minmax_image_path = r"C:\Users\songu\Downloads\outputs\PL4_ZScore_Normalized.png"
median_filter_image_path = r"C:\Users\songu\Downloads\outputs\PL4_ZScore_Median_Filtered.png"
gamma_corrected_image_path = r"C:\Users\songu\Downloads\outputs\PL4_Sobel_Edge_Detected.png"
l1_gaussianblur_sobel_gamma_path = r"C:\Users\songu\Downloads\outputs\PL4_Histogram_Equalized.png" # Yeni görüntü yolu

# Görüntüleri okuma
original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
minmax_img = cv2.imread(minmax_image_path, cv2.IMREAD_GRAYSCALE)
median_filter_img = cv2.imread(median_filter_image_path, cv2.IMREAD_GRAYSCALE)
gamma_corrected_img = cv2.imread(gamma_corrected_image_path, cv2.IMREAD_GRAYSCALE)
l1_gaussianblur_sobel_gamma_img = cv2.imread(l1_gaussianblur_sobel_gamma_path, cv2.IMREAD_GRAYSCALE)  # Yeni görüntü okuma

# Görüntüleri yan yana matplotlib ile gösterme
plt.figure(figsize=(25, 5))  # Yatayda 5 görüntü olacak şekilde genişletme

plt.subplot(1, 5, 1)
plt.imshow(original_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(minmax_img, cmap='gray')
plt.title('Z-Score Norm Applied')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(median_filter_img, cmap='gray')
plt.title('Median Filtered')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(gamma_corrected_img, cmap='gray')
plt.title('Sobel Edge Detected')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(l1_gaussianblur_sobel_gamma_img, cmap='gray')  # Yeni görüntü
plt.title('Histogram Equalized')
plt.axis('off')

plt.tight_layout()
plt.show()
