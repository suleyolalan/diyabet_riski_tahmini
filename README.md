# diyabet_riski_tahmini

Bu projede, bireylerin diyabet hastalığı riski taşıyıp taşımadığını tahmin etmek amacıyla çeşitli makine öğrenmesi algoritmaları karşılaştırmalı olarak analiz edilmiştir.

Çalışmada Pima Indian Diabetes veri seti kullanılmıştır. Veri ön işleme sürecinde:

Aykırı değerler temizlenmiş,

Veriler standartlaştırılmış,

Eğitim ve test setlerine bölünmüştür.

Sekiz farklı makine öğrenmesi modeli, çapraz doğrulama yöntemiyle eğitilmiş ve elde edilen doğruluk oranları karşılaştırılmıştır. En düşük performansa sahip olan Decision Tree algoritmasına hiperparametre optimizasyonu uygulanarak anlamlı bir performans iyileştirmesi sağlanmıştır.

Eğitim sürecinin ardından, geliştirilmiş model gerçekçi bir kullanıcı verisi ile test edilmiş ve pratikteki uygulanabilirliği gözlemlenmiştir.

Bu proje, makine öğrenmesi yöntemlerinin sağlık alanında tahmine dayalı karar destek sistemlerinde nasıl etkili bir şekilde kullanılabileceğini göstermektedir.

## 📂 Kullanılan Veri Seti
- Kaggle üzerinden alınan [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## 🛠️ Kullanılan Yöntemler
- Veri Temizleme
- Standartizasyon
- Train-Test Ayrımı
- Cross-Validation
- Hiperparametre Optimizasyonu (GridSearchCV)
- Yeni Veri ile Tahmin

## 🤖 Kullanılan Modeller
- LogisticRegression 
- DecisionTreeClassifier 
- KNeighborsClassifier
- GaussianNB
- SVC
- AdaBoostClassifier
- GradientBoostingClassifier
- RandomForestClassifier



