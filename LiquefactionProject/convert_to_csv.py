import pandas as pd

# Düzenlenmiş CSV dosyasını oku
df = pd.read_csv('duzenlenmis_veri.csv')

# Sütun isimlerini temizle
df.columns = df.columns.str.strip()

# Eksik değerleri kontrol et
print("Eksik değer sayısı:")
print(df.isnull().sum())

# Veri tiplerini kontrol et
print("\nVeri tipleri:")
print(df.dtypes)

# UTF-8 CSV olarak kaydet (virgülle ayırarak)
df.to_csv('duzenlenmis_veri_utf8.csv', index=False, encoding='utf-8', sep=',')

print("Dönüştürme tamamlandı.") 