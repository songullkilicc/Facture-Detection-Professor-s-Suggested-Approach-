import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology
from skimage.filters import sobel
import os

# Görsel yolları
image_paths = [
    r"C:\Users\songu\Downloads\images\kirik_6.png",
    r"C:\Users\songu\Downloads\images\kirik_3.png",
    r"C:\Users\songu\Downloads\images\kirik_2.png"
]

# Her bir görseli işlemeye başlıyoruz
for image_path in image_paths:
    # Görseli yükle
    label_image = io.imread(image_path)

    # Görsel RGBA formatındaysa, sadece RGB kanallarını kullan
    if label_image.ndim == 3 and label_image.shape[2] == 4:
        label_image = label_image[..., :3]  # Sadece RGB kanallarını al

    # Görseli gri tonlamaya çevir
    gray_image = color.rgb2gray(label_image)

    # Kontrastı arttırmak için normalize etme işlemi
    norm_image = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image))

    # Thresholding (parlak kemik alanlarını seçmek için)
    threshold_value = np.max(norm_image) * 0.65  # Kemik daha parlak olduğu için threshold'u biraz daha düşürdük
    bone_mask = norm_image > threshold_value

    # Yumuşatma işlemi (blur) ile gürültüyü azaltıyoruz
    smoothed_image = filters.gaussian(norm_image, sigma=1)

    # Sobel kenar tespiti uyguluyoruz
    edges = sobel(smoothed_image)

    # Maskeyi kullanarak kemiği beyaz, arka planı siyah yapıyoruz
    final_result = np.zeros_like(norm_image)  # Siyah bir görsel oluşturuyoruz
    final_result[bone_mask] = 1  # Kemik alanlarını beyaz yapıyoruz

    # Kenarları ekliyoruz (kemik sınırlarının daha belirgin olmasını sağlıyoruz)
    final_result[edges > 0.05] = 1  # Kenarları belirginleştiriyoruz

    # Dış kenarları biraz daha karartmak için bir işlem ekliyoruz
    final_result = (final_result > 0).astype(np.bool)  # final_result'i bool türüne dönüştürüyoruz
    final_result = morphology.remove_small_objects(final_result, min_size=500)  # Küçük nesneleri kaldırıyoruz

    # Alt kısmı kapsamak için dilate işlemi (büyütme)
    final_result = morphology.dilation(final_result, morphology.square(5))  # 5x5 boyutunda bir kernel ile büyütme

    # Sonuçları görselleştir
    plt.figure(figsize=(6, 6))
    plt.imshow(final_result, cmap='gray')
    plt.title(f"Kemiğin Tamamı ve Kenarları - {os.path.basename(image_path)}")
    plt.show()
