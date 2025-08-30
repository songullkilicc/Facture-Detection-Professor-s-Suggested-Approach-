import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gaussian Blur uygulama fonksiyonu
def apply_gaussian_blur(image_path, output_path):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Daha belirgin bir etkisi olması için daha büyük bir kernel (7x7)
    blurred_img = cv2.GaussianBlur(img, (7, 7), 0)

    # Çıktıyı belirlenen klasöre kaydetme
    cv2.imwrite(output_path, blurred_img)

    # Görüntüyü matplotlib ile gösterme
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('L1 Norm Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(blurred_img, cmap='gray')
    plt.title('Gaussian Blur Applied ')
    plt.axis('off')

    plt.show()

# Görüntü yolu ve çıktı yolu
input_image_path = "C:\\Users\\songu\\Downloads\\outputs\\L1_norm.png"
output_image_path = "C:\\Users\\songu\\Downloads\\outputs\\L1_gaussianblur.png"

# Gaussian Blur işlemi fonksiyonunu çağırma
apply_gaussian_blur(input_image_path, output_image_path)

print(f"Çıktı: {output_image_path} olarak kaydedildi.")

