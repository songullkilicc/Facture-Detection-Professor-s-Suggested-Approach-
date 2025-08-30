import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gamma Correction uygulama fonksiyonu
def apply_gamma_correction(image_path, output_path, gamma=1.5):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Normalize etme
    norm_img = img / 255.0
    
    # Gamma Correction uygulama
    gamma_corrected_img = np.power(norm_img, gamma) * 255
    gamma_corrected_img = np.clip(gamma_corrected_img, 0, 255).astype(np.uint8)
    
    # Sonuç görüntüsünü kaydetme
    cv2.imwrite(output_path, gamma_corrected_img)
    
    # Matplotlib ile gösterim
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Gamma Correction")
    plt.imshow(gamma_corrected_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Kullanım
input_image_path = r"C:\Users\songu\Downloads\outputs\L1_gaussianblur_sobel.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\L1_gaussianblur_sobel_gamma.png"

# Gamma Correction uygulama
apply_gamma_correction(input_image_path, output_image_path, gamma=1.5)

print(f"Gamma Correction sonrası görüntü {output_image_path} adresine kaydedildi.")
