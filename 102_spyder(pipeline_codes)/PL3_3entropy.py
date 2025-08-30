import cv2
import numpy as np
import matplotlib.pyplot as plt

# Entropy Filter uygulama fonksiyonu
def apply_entropy_filter(image_path, output_path, ksize=1):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Entropi filtresi uygulama
    entropy_img = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    entropy_img = np.abs(entropy_img)
    
    # Entropiyi normalize etme
    entropy_img = np.uint8(entropy_img / entropy_img.max() * 255)
    
    # Sonuç görüntüsünü kaydetme
    cv2.imwrite(output_path, entropy_img)
    
    # Matplotlib ile gösterim
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Entropy Filter")
    plt.imshow(entropy_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Kullanım
input_image_path = r"C:\Users\songu\Downloads\outputs\PL3_Adaptive_median.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL3_Adaptive_median_entropy.png"

# Entropy Filter uygulama
apply_entropy_filter(input_image_path, output_image_path, ksize=3)

print(f"Entropy Filter sonrası görüntü {output_image_path} adresine kaydedildi.")
