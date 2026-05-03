# 🛍️ Retail Clothing Price Prediction

Proyek machine learning untuk memprediksi harga pakaian retail menggunakan berbagai algoritma regresi dan interpretasi model berbasis SHAP.

---

## 📋 Deskripsi Proyek

Notebook ini membangun model prediksi harga pakaian berdasarkan atribut produk seperti merek, kategori, warna, ukuran, dan material. Tujuannya adalah menemukan model terbaik yang mampu memperkirakan harga secara akurat, serta memahami fitur mana yang paling berpengaruh terhadap harga menggunakan teknik explainability **SHAP**.

Dataset bersumber dari Kaggle: [`mrsimple07/clothes-price-prediction`](https://www.kaggle.com/datasets/mrsimple07/clothes-price-prediction)

---

## 📁 Struktur Proyek

```
├── SDPI_3_RetailPricing.ipynb   # Notebook utama
├── shap_summary.png             # Plot SHAP summary (dot)
├── shap_importance.png          # Plot SHAP feature importance (bar)
├── shap_persen.png              # Kontribusi fitur dalam persen
├── actual_vs_predicted.png      # Scatter plot actual vs predicted semua model
└── README.md
```

---

## 🗂️ Dataset

| Atribut | Keterangan |
|---|---|
| **Sumber** | Kaggle — `mrsimple07/clothes-price-prediction` |
| **File** | `clothes_price_prediction_data.csv` |
| **Target** | `Price` (harga pakaian) |
| **Fitur** | `Brand`, `Category`, `Color`, `Size`, `Material` |

### Fitur Kategorikal
- **Brand** — Merek pakaian
- **Category** — Jenis pakaian (e.g., shirt, pants, dress, dll.)
- **Color** — Warna produk
- **Size** — Ukuran (S, M, L, XL, dll.)
- **Material** — Bahan kain (cotton, polyester, dll.)

---

## ⚙️ Alur Pengerjaan

### 1. 📥 Load Data
Data dimuat langsung dari Kaggle menggunakan `kagglehub` dengan adapter Pandas.

### 2. 🔍 Analisis Data (EDA)
- Pengecekan shape, kolom, dan missing values
- Deteksi outlier dengan metode **IQR** (Interquartile Range)
  - Batas bawah: `Q1 - 1.5 × IQR`
  - Batas atas: `Q3 + 1.5 × IQR`
- Visualisasi distribusi harga menggunakan Boxplot
- Analisis nilai unik tiap fitur kategorikal
- Mean price per kategori
- Korelasi fitur dengan target `Price` setelah Label Encoding

### 3. 🔧 Preprocessing
- **Label Encoding** pada semua fitur kategorikal (`Brand`, `Category`, `Color`, `Size`, `Material`)
- Split data: **80% training / 20% testing** (`random_state=42`)

### 4. 🤖 Training & Evaluasi Model
Empat model dilatih dan dievaluasi:

| Model | Keterangan |
|---|---|
| **Linear Regression** | Model baseline regresi linier |
| **Decision Tree** | Pohon keputusan tanpa ensemble |
| **Random Forest** | Ensemble dari banyak pohon keputusan |
| **XGBoost** | Gradient boosting yang dioptimalkan |

Metrik evaluasi yang digunakan:
- **MAE** (Mean Absolute Error) — rata-rata selisih absolut prediksi vs aktual
- **RMSE** (Root Mean Squared Error) — akar rata-rata kuadrat error
- **R²** (R-squared) — proporsi variansi yang dijelaskan model (semakin mendekati 1, semakin baik)

### 5. 📊 Interpretasi Model (SHAP)
Menggunakan library **SHAP** (SHapley Additive exPlanations) untuk menjelaskan kontribusi setiap fitur terhadap prediksi harga:
- **SHAP Summary Plot** (dot) — distribusi nilai SHAP per fitur
- **SHAP Feature Importance** (bar) — rata-rata kontribusi absolut tiap fitur
- **Kontribusi Fitur (%)** — persentase pengaruh relatif tiap fitur

### 6. 📋 Perbandingan Harga Aktual vs Prediksi
Tabel perbandingan antara harga asli dan harga prediksi model terbaik, lengkap dengan:
- Selisih absolut
- Selisih dalam persen (%)
- Ringkasan akurasi (prediksi terlalu tinggi/rendah, prediksi tepat ±5%)

---

## 📦 Dependencies

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

| Library | Kegunaan |
|---|---|
| `kagglehub` | Download dataset dari Kaggle |
| `pandas`, `numpy` | Manipulasi dan analisis data |
| `matplotlib`, `seaborn` | Visualisasi data |
| `scikit-learn` | Preprocessing, model ML, dan evaluasi |
| `xgboost` | Model gradient boosting |
| `shap` | Interpretasi/explainability model |

---

## 🚀 Cara Menjalankan

1. Clone repo ini:
   ```bash
   git clone https://github.com/username/retail-pricing.git
   cd retail-pricing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Pastikan kamu sudah setup Kaggle API credentials (`~/.kaggle/kaggle.json`)

4. Buka dan jalankan notebook:
   ```bash
   jupyter notebook SDPI_3_RetailPricing.ipynb
   ```

---

## 📈 Hasil & Performa Model

Model dievaluasi menggunakan metrik MAE, RMSE, dan R². Model terbaik dipilih otomatis berdasarkan nilai **R² tertinggi**, kemudian digunakan untuk analisis SHAP dan perbandingan prediksi vs aktual.

> **Catatan:** Karena dataset ini bersifat sintetik (data buatan), nilai R² yang tinggi pada model berbasis pohon (Random Forest / XGBoost) wajar terjadi, namun perlu diwaspadai potensi overfitting jika diaplikasikan ke data dunia nyata.

### Interpretasi Metrik

| Metrik | Artinya |
|---|---|
| **MAE rendah** | Rata-rata prediksi tidak jauh dari harga sebenarnya |
| **RMSE rendah** | Error besar (outlier prediksi) lebih jarang terjadi |
| **R² mendekati 1** | Model mampu menjelaskan variasi harga dengan baik |

---

## 🧠 Insight dari SHAP

SHAP memungkinkan kita melihat *mengapa* model membuat prediksi tertentu. Beberapa insight umum yang bisa diperoleh dari proyek ini:

- Fitur mana yang paling dominan memengaruhi harga (e.g., Brand vs Material)
- Apakah pengaruh fitur bersifat positif atau negatif terhadap harga
- Seberapa konsisten pengaruh suatu fitur di berbagai produk

---

## 📌 Catatan

- Dataset bersifat **sintetik** — distribusi harga mungkin seragam per kategori, sehingga model tree-based bisa mencapai R² sangat tinggi
- Label Encoding digunakan sebagai pendekatan sederhana; untuk produksi, **One-Hot Encoding** atau **Target Encoding** bisa dipertimbangkan
- SHAP dihitung pada 200 sampel test pertama untuk efisiensi komputasi

