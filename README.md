# DSF-Dibimbing
# Student Score Prediction

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk memprediksi nilai ujian siswa berdasarkan jumlah jam belajar menggunakan algoritma regresi. Dataset yang digunakan berisi dua fitur utama:
- **Hours**: Jumlah jam belajar siswa.
- **Scores**: Nilai ujian siswa.

Beberapa teknik machine learning yang digunakan dalam proyek ini:
1. **Linear Regression**
2. **Decision Tree Regression**
3. **Random Forest Regression**

## 📂 Struktur Direktori
```
|-- Supervised_Regression_StudentScore_Prediction.ipynb  # Notebook utama proyek
|-- student_scores.csv                                  # Dataset
|-- README.md                                           # Dokumentasi proyek
```

## 📊 Eksplorasi Data
Dataset diperiksa dengan beberapa langkah berikut:
- **Menampilkan beberapa data awal** menggunakan `data.head()`.
- **Menampilkan informasi dataset** (`data.info()`).
- **Melihat statistik dasar dataset** (`data.describe()`).
- **Mengecek duplikasi data** (`df.duplicated().sum()`).
- **Mengecek missing values** (`df.isna().sum()`).
- **Analisis Outlier** menggunakan boxplot.

## 🏗️ Feature Engineering
- Dataset ini tidak memiliki data kategori, sehingga tidak perlu encoding.
- Tidak ditemukan data duplikat maupun missing values.
- Tidak ada outlier yang signifikan, sehingga data dapat langsung digunakan untuk pelatihan model.

## 🔀 Pembagian Data
Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan `train_test_split()`:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

## 🤖 Model Machine Learning
### 1️⃣ Linear Regression
Linear Regression digunakan untuk memodelkan hubungan antara jumlah jam belajar dan nilai ujian.
```python
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```
### 2️⃣ Decision Tree Regression
Algoritma Decision Tree digunakan untuk membuat model yang lebih fleksibel.
```python
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
```
### 3️⃣ Random Forest Regression
Random Forest digunakan untuk meningkatkan akurasi prediksi dengan metode ensemble.
```python
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)
```

## 📈 Evaluasi Model
Evaluasi dilakukan menggunakan nilai **R-squared** (`r2_score`) untuk mengukur seberapa baik model mampu menjelaskan variabilitas data.
```python
from sklearn.metrics import r2_score
rsq_lr = r2_score(y_test, lr_model.predict(X_test))
rsq_dt = r2_score(y_test, dt_model.predict(X_test))
rsq_rf = r2_score(y_test, rf_model.predict(X_test))
```
### 🔍 Hasil Evaluasi
| Model | R-squared |
|--------|-----------|
| Linear Regression | 0.95 |
| Decision Tree | 0.81 |
| Random Forest | 0.80 |

## 📊 Visualisasi Perbandingan Model
```python
import matplotlib.pyplot as plt
models = ['Linear Regression', 'Decision Tree', 'Random Forest']
r_squared = [rsq_lr, rsq_dt, rsq_rf]
plt.bar(models, r_squared, color=['blue', 'green', 'orange'])
plt.title('Model Comparison: R-squared Scores')
plt.xlabel('Model')
plt.ylabel('R-squared Score')
plt.show()
```

## 📌 Kesimpulan
- **Linear Regression memiliki performa terbaik** dengan nilai R-squared sebesar **0.95**.
- **Decision Tree dan Random Forest memiliki performa lebih rendah** dibandingkan Linear Regression pada dataset ini.
- Model dapat dikembangkan lebih lanjut dengan melakukan **feature engineering** atau menggunakan dataset yang lebih besar.

## 📌 Cara Menjalankan Proyek
1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/StudentScorePrediction.git
   ```
2. Buka dan jalankan notebook di Jupyter Notebook atau Google Colab.
3. Pastikan semua library yang diperlukan sudah terinstal:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Jalankan setiap sel dalam notebook untuk melihat hasil prediksi.

🚀 **Selamat mencoba!**

