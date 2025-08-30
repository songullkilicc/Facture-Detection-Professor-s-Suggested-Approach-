import cv2
import matplotlib.pyplot as plt

# Histogram eşikleme uygulama fonksiyonu
def apply_histogram_equalization(image_path, output_path):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Histogram eşikleme
    hist_eq_img = cv2.equalizeHist(img)
    
    # Sonuç görüntüsünü kaydetme
    cv2.imwrite(output_path, hist_eq_img)
    
    # Matplotlib ile gösterim
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Orijinal Görüntü")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Histogram Equalization")
    plt.imshow(hist_eq_img, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Kullanım
input_image_path =  r"C:\Users\songu\Downloads\outputs\PL2_MinMax_medianfilter_laplacian.png"
output_image_path = r"C:\Users\songu\Downloads\outputs\PL2_MinMax_medianfilter_laplacian_histeq.png"

# Histogram eşikleme uygulama
apply_histogram_equalization(input_image_path, output_image_path)

print(f"Histogram Eşitleme sonrası görüntü {output_image_path} adresine kaydedildi.")
