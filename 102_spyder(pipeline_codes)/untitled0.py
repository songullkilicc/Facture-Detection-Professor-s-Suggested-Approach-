rimport cv2
import numpy as np
import pybrisque
import matplotlib.pyplot as plt
import pandas as pd

# BRISQUE skorunu hesaplamak için fonksiyon
def calculate_brisque(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    return pybrisque.score(img)

# Test edilecek görüntülerin ve pipeline çıktılarının yolları
images = [
    r"C:\Users\songu\Downloads\outputs\PL2_MinMax_medianfilter_laplacian_histeq.png",
    r"C:\Users\songu\Downloads\outputs\PL3_Adaptive_median_entropy_brightness.png",
    r"C:\Users\songu\Downloads\outputs\PL4_Histogram_Equalized.png"
]

pipelines = {
    "Original": "",
    "L2 Norm + Mean Filter": "_L2_Mean.png",
    "Sobel Edge Detection": "_Sobel.png",
    "Histogram Equalization": "_Histogram_Equalized.png"
}

# Sonuçları kaydetmek için bir tablo
results = {
    "Image": [],
    "Pipeline": [],
    "BRISQUE Score": []
}

# Her görüntü ve pipeline için BRISQUE skorlarını hesapla
for image_path in images:
    base_name = image_path.split("\\")[-1]  # Görüntü dosyasının adı
    for pipeline_name, suffix in pipelines.items():
        # Pipeline çıktısının yolu
        output_path = image_path.replace(".png", suffix) if suffix else image_path
        
        # BRISQUE skorunu hesapla
        score = calculate_brisque(output_path)
        if score is not None:
            results["Image"].append(base_name)
            results["Pipeline"].append(pipeline_name)
            results["BRISQUE Score"].append(score)

# Pandas DataFrame ile sonuçları göster
df_results = pd.DataFrame(results)

# Tabloyu ekrana yazdır
print(df_results)

# Görselleştirme
plt.figure(figsize=(12, 8))
for image_name in df_results["Image"].unique():
    image_data = df_results[df_results["Image"] == image_name]
    plt.plot(image_data["Pipeline"], image_data["BRISQUE Score"], marker='o', label=image_name)

plt.xlabel("Pipeline")
plt.ylabel("BRISQUE Score")
plt.title("BRISQUE Scores for Different Pipelines and Images")
plt.xticks(rotation=45)
plt.legend(title="Images")
plt.tight_layout()
plt.show()


