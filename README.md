# Dengesiz Türkçe E-Ticaret Yorumlarında Duygu Analizi: Klasik Makine Öğrenmesi, BERTurk ve Hata Kategorisi Tabanlı Bir Karşılaştırma

Bu proje, ciddi sınıf dengesizliği içeren Türkçe e-ticaret ürün yorumlarının duygu analizi (olumlu/olumsuz sınıflandırma) problemini ele almaktadır. Klasik makine öğrenmesi modelleri ile Transformer tabanlı modeller sistematik olarak karşılaştırılmıştır.

## Veri Seti

- **Kaynak:** [fthbrmnby/turkish_product_reviews](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews) (Hugging Face)
- **Boyut:** 235.165 yorum
- **Sınıf dağılımı:** %93,7 olumlu, %6,3 olumsuz (ciddi sınıf dengesizliği)

## Kullanılan Yöntemler

**Özellik çıkarımı:**
- Word TF-IDF (kelime tabanlı, 1-2 gram)
- Char n-gram TF-IDF (karakter tabanlı, 3-5 gram)
- Word + Char hibrit temsil

**Sınıf dengeleme stratejileri:**
- SMOTE (sentetik örnekleme)
- Class weight (sınıf ağırlığı)
- Karar eşiği optimizasyonu

**Klasik makine öğrenmesi modelleri:**
- Linear SVM
- Lojistik Regresyon
- SGDClassifier
- Naive Bayes

**Transformer tabanlı modeller:**
- BERTurk (dbmdz/bert-base-turkish-uncased)
- XLM-RoBERTa (xlm-roberta-base)

## Değerlendirme

Model başarımı; Accuracy, Precision, Recall, F1 (sınıf bazında) ve Macro F1 metrikleriyle değerlendirilmiştir. Ayrıca:
- **Hata analizi:** Yanlış sınıflandırılan olumsuz yorumlar; bağlamsal ifade, olumsuzlama, yazım bozukluğu, kısa yorum, ironi ve karma duygu kategorilerine ayrılarak incelenmiştir.
- **İstatistiksel anlamlılık testleri:** McNemar testi (ikili model karşılaştırmaları) ve Friedman testi (klasik modeller arası genel karşılaştırma).

## Dosyalar

- `*.ipynb` — Tüm veri ön işleme, model eğitimi, değerlendirme ve analiz kodlarını içeren Jupyter/Colab notebook'u

## Ortam

Notebook, Google Colab üzerinde T4 GPU ile çalıştırılmak üzere hazırlanmıştır.

## Yazar

Melek Aday — Karadeniz Teknik Üniversitesi, Of Teknoloji Fakültesi, Yazılım Mühendisliği
Danışman: Dr. Öğr. Üyesi Sefa Aras



## Durum

🚧 Bu proje aktif olarak geliştirilmektedir 
