import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# Veri seti bilgilerini göster
print("\nVeri Seti Bilgileri:")
print(df.info())
print("\nİlk 5 satır:")
print(df.head())

# Özellikler ve hedef değişkeni belirle
features = [
    'derinlik',      # Derinlik
    'SPT-N',         # Standart Penetrasyon Testi değeri
    'gamma',         # Birim hacim ağırlık
    'sigmav\'',      # Efektif düşey gerilme
    'wn',            # Doğal su içeriği
    'PI',            # Plastisite indeksi
    'LL',            # Likit limit
    'PL',            # Plastik limit
    'IDO (FC)',      # İnce daneli oranı
    'd',             # Doygunluk derecesi
    'CN',            # Düzeltme faktörü
    'N1.60',         # Düzeltilmiş SPT değeri
    'rd',            # Gerilme azaltma faktörü
    'CSR',           # Döngüsel gerilme oranı
    'CRR',           # Döngüsel direnç oranı
    'Mw'             # Deprem büyüklüğü
]

target = 'F.S'  # Güvenlik faktörü (1'den büyükse sıvılaşma yok, 1'den küçükse sıvılaşma var)

# Veriyi hazırla
X = df[features]
y = df[target]

# Hedef değişkeni binary yap (1: sıvılaşma var, 0: sıvılaşma yok)
y = (y < 1).astype(int)  # Güvenlik faktörü 1'den küçükse sıvılaşma var

# Eksik değerleri kontrol et
print("\nEksik değer sayısı:")
print(X.isnull().sum())

# Eksik değerleri doldur
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Sınıf dağılımını kontrol et
print("\nSınıf dağılımı:")
print(pd.Series(y).value_counts(normalize=True))

# Özellikleri ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Sınıf ağırlıklarını hesapla
class_weights = {0: 1, 1: sum(y==0)/sum(y==1)}

# Modelleri tanımla
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weights,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=sum(y==0)/sum(y==1),
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        class_weight=class_weights,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=1.0,
        class_weight=class_weights,
        random_state=42
    )
}

# Cross-validation ile modelleri değerlendir
results = {}
for name, model in models.items():
    print(f"\n{name} modeli değerlendiriliyor...")
    
    # Cross-validation skorları
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
    
    # Confusion matrix için tahminler
    y_pred = cross_val_predict(model, X_scaled, y, cv=5)
    
    results[name] = {
        'F1 (mean)': cv_scores.mean(),
        'F1 (std)': cv_scores.std(),
        'Accuracy': accuracy_score(y, y_pred),
        'Confusion Matrix': confusion_matrix(y, y_pred)
    }
    
    print(f"F1 Skoru (ortalama): {results[name]['F1 (mean)']:.4f} (+/- {results[name]['F1 (std)']:.4f})")
    print(f"Doğruluk: {results[name]['Accuracy']:.4f}")
    print("Confusion Matrix:")
    print(results[name]['Confusion Matrix'])

# En iyi modeli seç (ortalama F1 skoruna göre)
best_model_name = max(results.items(), key=lambda x: x[1]['F1 (mean)'])[0]
best_model = models[best_model_name]

print(f"\nEn iyi model: {best_model_name}")
print(f"F1 Skoru: {results[best_model_name]['F1 (mean)']:.4f}")

# En iyi modeli tüm veri ile eğit ve kaydet
best_model.fit(X_scaled, y)
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'feature_names.pkl')

print("\nEn iyi model ve ölçeklendirici kaydedildi. Görselleştirmeler için visualize_results.py dosyasını çalıştırabilirsiniz.") 