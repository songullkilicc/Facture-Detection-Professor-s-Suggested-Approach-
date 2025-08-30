import cv2
import numpy as np
import matplotlib.pyplot as plt

# L1 Norm uygulama fonksiyonu
def l1_norm(image_path, output_path):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # L1 norm hesaplama
    norm = np.sum(np.abs(img))

    # L1 normdan sonra, normalizasyon: Her pikselin değerini 0-255 arasında getirmek için
    norm_img = (img / norm) * 255
    
    # Düzgün görüntüleme için min-max normalization (0-255 aralığında)
    norm_img = cv2.normalize(norm_img, None, 0, 255, cv2.NORM_MINMAX)

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
    plt.title('L1 Norm Applied')
    plt.axis('off')

    plt.show()


# Görüntü yolu ve çıktı yolu
input_image_path = r"C:\Users\songu\Downloads\images\kirik_6.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\L1_norm.png"

# L1 norm uygulama fonksiyonunu çağırma
l1_norm(input_image_path, output_image_path)

print(f"Çıktı: {output_image_path} olarak kaydedildi.")