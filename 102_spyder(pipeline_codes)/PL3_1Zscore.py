import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_zscore_norm(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mean = np.mean(img)
    std = np.std(img)
    zscore_img = ((img - mean) / std) * 128 + 128  # Normalize edip 0-255 aralığına getirme
    zscore_img = np.clip(zscore_img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, zscore_img)

    # Görselleştirme
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(zscore_img, cmap='gray')
    plt.title('Z-Score Normalized')
    plt.axis('off')

    plt.show()

# Kullanım
input_image_path = r"C:\Users\songu\Downloads\kirik_3.png"
zscore_output_path = r"C:\Users\songu\Downloads\outputs\PL3_Zscore_norm.png"

apply_zscore_norm(input_image_path, zscore_output_path)
