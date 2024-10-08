# 1. Mengimpor Library yang Diperlukan
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 2. Membaca File CSV dengan Pemisah Titik Koma (;)
# Gantilah 'lulus.csv' dengan nama file CSV-mu yang sebenarnya
df = pd.read_csv('lulus.csv', sep=',')  # Menggunakan pemisah koma untuk file CSV

# 3. Menampilkan Data Sebelum Preprocessing
print("Data Sebelum Preprocessing:")
print(df)

# 4. Mengatasi Nilai yang Hilang (Missing Values) dengan SimpleImputer
imputer = SimpleImputer(strategy='mean')  # Menggunakan rata-rata untuk mengisi nilai yang hilang

# Mengisi nilai hilang untuk kolom yang relevan
df[['IPK', 'Pelatihan Pengembangan Diri', 'Prestasi', 'Forum Komunikasi Kuliah', 'Kegiatan Organisasi']] = imputer.fit_transform(
    df[['IPK', 'Pelatihan Pengembangan Diri', 'Prestasi', 'Forum Komunikasi Kuliah', 'Kegiatan Organisasi']]
)

# 5. Encoding Data Kategori
# Label Encoding untuk kolom 'Lulus Cepat'
label_encoder = LabelEncoder()
df['Lulus Cepat'] = label_encoder.fit_transform(df['Lulus Cepat'])

# 6. Menampilkan Data Setelah Preprocessing
print("\nData Setelah Preprocessing:")
print(df)

# 7. Menyimpan Hasil Preprocessing ke dalam File CSV Baru
df.to_csv('preprocessing_output.csv', index=False)

print("\nData berhasil diproses dan disimpan ke 'preprocessing_output.csv'.")