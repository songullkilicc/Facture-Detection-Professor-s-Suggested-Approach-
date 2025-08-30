import cv2
import matplotlib.pyplot as plt

def apply_adaptive_median(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    adaptive_median_img = cv2.medianBlur(img, 3)  # Adaptive kernel size 5x5
    cv2.imwrite(output_path, adaptive_median_img)

    # Görselleştirme
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Z-Score Normalized')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(adaptive_median_img, cmap='gray')
    plt.title('Adaptive Median Applied')
    plt.axis('off')

    plt.show()

# Kullanım
zscore_output_path = r"C:\Users\songu\Downloads\outputs\PL3_Zscore_norm.png"
adaptive_median_output_path = r"C:\Users\songu\Downloads\outputs\PL3_Adaptive_median.png"

apply_adaptive_median(zscore_output_path, adaptive_median_output_path)