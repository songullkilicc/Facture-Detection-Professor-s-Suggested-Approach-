import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parlaklık ayarlama fonksiyonu
def apply_brightness_adjustment(image_path, output_path, brightness=50):
    # Görüntüyü okuma
    img = cv2.imread(image_path)
    
    # Parlaklık ayarlaması (bunu artırarak veya azaltarak değiştirebilirsiniz)
    adjusted_img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)
    
    # Sonuç görüntüsünü kaydetme
    cv2.imwrite(output_path, adjusted_img)
    
    # Matplotlib ile gösterim
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Parlaklık Ayarlanmış Görüntü")
    plt.imshow(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Kullanım
input_image_path = r"C:\Users\songu\Downloads\outputs\PL3_Adaptive_median_entropy.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL3_Adaptive_median_entropy_brightness.png"

# Parlaklık ayarlaması uygulama
apply_brightness_adjustment(input_image_path, output_image_path, brightness=50)

print(f"Parlaklık ayarlama sonrası görüntü {output_image_path} adresine kaydedildi.")
