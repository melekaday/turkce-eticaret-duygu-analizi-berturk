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
- **Ablasyon çalışması:** Word + Char hibrit temsilin katkısını ölçmek amacıyla ablasyon tablosu (`word_char_hybrid_ablasyon`) oluşturulmuştur.

## Dosya Yapısı
├── *.ipynb # Veri ön işleme, model eğitimi, değerlendirme ve analiz kodları
│
├── bert_checkpoints/ # BERTurk eğitim çıktıları (model ağırlıkları)
├── xlmr_checkpoints/ # XLM-RoBERTa eğitim çıktıları (model ağırlıkları)
│
├── final_smote_lr_model.pkl # SMOTE + Lojistik Regresyon final modeli
├── final_word_vectorizer.pkl # Word TF-IDF vectorizer
├── final_char_vectorizer.pkl # Char n-gram TF-IDF vectorizer
│
├── proba_lr_val.npy / proba_lr_test.npy # Lojistik Regresyon tahmin olasılıkları
├── proba_berturk_val.npy / proba_berturk_test.npy # BERTurk tahmin olasılıkları
├── proba_xlmr_val.npy / proba_xlmr_test.npy # XLM-RoBERTa tahmin olasılıkları
├── proba_bert4_val.npy / proba_bert4_test.npy # BERTurk (4. konfigürasyon) tahmin olasılıkları
├── y_pred_xlmr.npy # XLM-RoBERTa sınıf tahminleri
│
├── klasik_modeller_karsilastirma # Klasik ML modelleri karşılaştırma sonuçları
├── word_char_hybrid_ablasyon # Word+Char hibrit ablasyon tablosu
├── hata_analizi_bert # BERT hata analizi (kategori bazlı)
├── hata_analizi_kategoriler # Genel hata kategorileri analizi
├── confusion_matrix_bert # BERT karışıklık matrisi görseli
│
├── stageA_check → stageH_stacking # Aşama aşama işlem/analiz adımları
│ (veri kontrolü, transformer olasılık çıktıları, füzyon, uzunluk analizi,
│ decoupled değerlendirme, diagnostik, stacking sonuçları)


## Ortam

Notebook, Google Colab üzerinde T4 GPU ile çalıştırılmak üzere hazırlanmıştır.

> **Not:** Model checkpoint dosyaları büyük boyutlu olduğu için Git LFS ile yönetilmektedir. Repoyu klonlarken `git lfs install` çalıştırılması önerilir. Optimizer durum dosyaları (yalnızca eğitime devam etmek için gerekli, inference için gerekli değildir) depoya dahil edilmemiştir.

## Yazar

Melek Aday — Karadeniz Teknik Üniversitesi, Of Teknoloji Fakültesi, Yazılım Mühendisliği
Danışman: Dr. Öğr. Üyesi Sefa Aras


