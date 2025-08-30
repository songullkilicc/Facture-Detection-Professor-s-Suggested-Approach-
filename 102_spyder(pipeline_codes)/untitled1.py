from skimage.color import rgb2gray

# Görseli gri tonlamaya dönüştür
def remove_alpha_channel(image):
    if image.shape[2] == 4:  # Eğer görüntüde 4 kanal (RGBA) varsa
        image = image[:, :, :3]  # Sadece RGB kanallarını al
    return image

# Patch'leri test etme
def test_patches_with_erosion(image, label, patch_size=16, erosion_size=3, threshold=0.5):
    # Şeffaflık kanalını kaldırarak gri tonlamaya dönüştür
    image = remove_alpha_channel(image)
    image_gray = rgb2gray(image)
    
    # Label üzerinde erosion
    eroded_label = apply_erosion(label, size=erosion_size)
    eroded_label = eroded_label > 0  # Etiketli alanları filtrele
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Eroded Label")
    plt.imshow(eroded_label, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Original Image")
    plt.imshow(image_gray, cmap="gray")
    plt.show()

    results = []
    for x in range(eroded_label.shape[0]):
        for y in range(eroded_label.shape[1]):
            if eroded_label[x, y] > 0:  # Eğer erosion sonrası alan varsa
                patch = extract_patch(image_gray, x, y, patch_size)  # Patch çıkar
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    continue  # Patch küçükse işlemi atla
                try:
                    encoded_patch = encode_patch_with_hog(patch)    # HoG ile kodla
                    # Basit test: Patch ile label karşılaştırma
                    match_score = np.linalg.norm(encoded_patch - eroded_label[x, y])
                    result = 1 if match_score < threshold else 0
                    results.append((x, y, result))
                except ValueError as e:
                    print(f"Patch at ({x}, {y}) skipped: {e}")
                    continue

    print("Test Sonuçları:", results)

# Görsellerle çalıştırma (örnek olarak image1, label1, vb. yerine doğru görsel ve label verilerini kullanın)
test_patches_with_erosion(image1, label1, patch_size=16)
test_patches_with_erosion(image2, label2, patch_size=16)
test_patches_with_erosion(image3, label3, patch_size=16)
