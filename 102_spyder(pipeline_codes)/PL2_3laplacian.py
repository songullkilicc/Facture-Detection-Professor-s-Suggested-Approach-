import cv2
import numpy as np
import matplotlib.pyplot as plt

# Laplacian uygulama fonksiyonu
def apply_laplacian(image_path, output_path, ksize=3):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Laplacian filtresi uygulama
    laplacian_img = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
    laplacian_img = np.clip(np.abs(laplacian_img), 0, 255).astype(np.uint8)
    
    # Sonuç görüntüsünü kaydetme
    cv2.imwrite(output_path, laplacian_img)
    
    # Matplotlib ile gösterim
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Laplacian ")
    plt.imshow(laplacian_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Kullanım
input_image_path = r"C:\Users\songu\Downloads\outputs\PL2_MinMax_medianfilter.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL2_MinMax_medianfilter_laplacian.png"

# Laplacian uygulama
apply_laplacian(input_image_path, output_image_path, ksize=3)

print(f"Laplacian sonrası görüntü {output_image_path} adresine kaydedildi.")