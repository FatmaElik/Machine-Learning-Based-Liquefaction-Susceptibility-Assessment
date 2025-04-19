import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Görselleştirme ayarları
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Veri setini yükle (farklı encoding'leri dene)
try:
    df = pd.read_csv('duzenlenmis_veri.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('duzenlenmis_veri.csv', encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv('duzenlenmis_veri.csv', encoding='cp1254')

# Temel bilgileri göster
print("\nVeri Seti Bilgileri:")
print(df.info())
print("\nİlk 5 satır:")
print(df.head())
print("\nTemel İstatistikler:")
print(df.describe())

# Eksik değerleri kontrol et
print("\nEksik Değer Sayısı:")
print(df.isnull().sum())

# Eksik değerlerin görselleştirilmesi
plt.figure(figsize=(15, 8))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Eksik Değerlerin Dağılımı')
plt.tight_layout()
plt.savefig('missing_values.png')
plt.close()

# Sayısal değişkenlerin histogramları
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'{col} Dağılımı')
    plt.tight_layout()
    plt.savefig(f'histogram_{col}.png')
    plt.close()

# Boxplot'lar
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numeric_cols])
plt.title('Sayısal Değişkenlerin Boxplot Grafikleri')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplot_numeric.png')
plt.close()

# Korelasyon matrisi
plt.figure(figsize=(15, 12))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korelasyon Matrisi')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Hedef değişken (F.S) ile diğer değişkenlerin ilişkisi
target = 'F.S'
for col in numeric_cols:
    if col != target:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=col, y=target)
        plt.title(f'{col} vs {target}')
        plt.tight_layout()
        plt.savefig(f'scatter_{col}_vs_{target}.png')
        plt.close()

# Derinlik bazında analiz
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='derinlik', y='F.S')
plt.title('Derinliğe Göre Güvenlik Faktörü Değişimi')
plt.xlabel('Derinlik (m)')
plt.ylabel('Güvenlik Faktörü (F.S)')
plt.grid(True)
plt.tight_layout()
plt.savefig('depth_vs_safety_factor.png')
plt.close()

# SPT-N değerlerinin derinlikle değişimi
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='derinlik', y='SPT-N')
plt.title('Derinliğe Göre SPT-N Değerlerinin Değişimi')
plt.xlabel('Derinlik (m)')
plt.ylabel('SPT-N')
plt.grid(True)
plt.tight_layout()
plt.savefig('depth_vs_spt.png')
plt.close()

# Sıvılaşma potansiyeli analizi
df['Liquefaction_Potential'] = (df['F.S'] < 1).astype(int)
liquefaction_summary = df.groupby('Liquefaction_Potential').size()
print("\nSıvılaşma Potansiyeli Dağılımı:")
print(liquefaction_summary)

# Sıvılaşma potansiyeli olan ve olmayan noktaların derinlik dağılımı
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Liquefaction_Potential', y='derinlik')
plt.title('Sıvılaşma Potansiyeline Göre Derinlik Dağılımı')
plt.xlabel('Sıvılaşma Potansiyeli (0: Yok, 1: Var)')
plt.ylabel('Derinlik (m)')
plt.tight_layout()
plt.savefig('liquefaction_depth_distribution.png')
plt.close()

print("\nEDA tamamlandı. Görseller kaydedildi.")
