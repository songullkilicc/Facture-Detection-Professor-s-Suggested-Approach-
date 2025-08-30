import cv2
import numpy as np
import matplotlib.pyplot as plt

# Sobel Edge Detection uygulama fonksiyonu
def apply_sobel_edge_detection(image_path, output_path):
    # Görüntüyü okuma
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Sobel Edge Detection işlemi (X ve Y yönündeki kenarları tespit etme)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # X yönünde kenar
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Y yönünde kenar
    
    # X ve Y yönündeki kenarları birleştirerek tam kenar tespiti yapmak
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    
    # Sobel sonucu 0-255 aralığına normalize etme
    sobel_edges = np.uint8(np.absolute(sobel_edges))

    # Çıktıyı belirlenen klasöre kaydetme
    cv2.imwrite(output_path, sobel_edges)

    # Görüntüyü matplotlib ile gösterme
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Gaussian Blurred Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sobel_edges, cmap='gray')
    plt.title('Sobel Edge Detection Applied')
    plt.axis('off')

    plt.show()

# Görüntü yolu ve çıktı yolu
input_image_path = "C:\\Users\\songu\\Downloads\\outputs\\L1_gaussianblur.png"
output_image_path = "C:\\Users\\songu\\Downloads\\outputs\\L1_gaussianblur_sobel.png"

# Sobel Edge Detection fonksiyonunu çağırma
apply_sobel_edge_detection(input_image_path, output_image_path)

print(f"Çıktı: {output_image_path} olarak kaydedildi.")