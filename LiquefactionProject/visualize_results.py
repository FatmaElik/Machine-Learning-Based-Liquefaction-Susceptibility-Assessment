import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, roc_auc_score, accuracy_score, f1_score
import joblib

# Görselleştirme ayarları
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Modelleri ve verileri yükle
try:
    models = joblib.load('trained_models.pkl')
    X_test = joblib.load('X_test.pkl')
    y_test = joblib.load('y_test.pkl')
except:
    print("Önce liquefaction_analysis.py dosyasını çalıştırın!")
    exit()

# ROC Eğrileri
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrileri')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

# Precision-Recall Eğrileri
plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'{name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Eğrileri')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('precision_recall_curves.png')
plt.close()

# Detaylı Confusion Matrix'ler
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'{name} - Ham Confusion Matrix')
    ax1.set_ylabel('Gerçek Değer')
    ax1.set_xlabel('Tahmin Edilen Değer')
    
    # Plot normalized confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
    ax2.set_title(f'{name} - Normalize Edilmiş Confusion Matrix')
    ax2.set_ylabel('Gerçek Değer')
    ax2.set_xlabel('Tahmin Edilen Değer')
    
    # Add class labels
    for ax in [ax1, ax2]:
        ax.set_xticklabels(['Sıvılaşma Yok', 'Sıvılaşma Var'])
        ax.set_yticklabels(['Sıvılaşma Yok', 'Sıvılaşma Var'])
    
    plt.tight_layout()
    plt.savefig(f'detailed_confusion_matrix_{name}.png')
    plt.close()
    
    # Calculate and print metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{name} Modeli için Detaylı Metrikler:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive: {tp}")
    print(f"True Negative: {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")

# Özellik Önemi (Random Forest için)
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'Özellik': X_test.columns,
        'Önem': rf_model.feature_importances_
    }).sort_values('Önem', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Önem', y='Özellik', data=feature_importance)
    plt.title('Özellik Önemi (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Model Performans Karşılaştırması
metrics = ['AUC', 'Accuracy', 'F1']
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'AUC': roc_auc_score(y_test, y_prob),
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred, zero_division=0)
    }

plt.figure(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.2

for i, (name, result) in enumerate(results.items()):
    plt.bar(x + i*width, [result[metric] for metric in metrics], width, label=name)

plt.xlabel('Metrikler')
plt.ylabel('Skor')
plt.title('Model Performans Karşılaştırması')
plt.xticks(x + width, metrics)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

print("\nGörselleştirmeler başarıyla oluşturuldu ve kaydedildi.") 