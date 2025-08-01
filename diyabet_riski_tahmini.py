# İzlediğim sıralama şu şekilde:

# 1- import libraries : öncelikle gerekli kütüphaneleri içeri aktarıyorum
# 2- import data and EDA : verinin içeri aktarılması ve keşifsel ver analizi
# 3- outlier detection : outlier tespiti
# 4- Train test split : verisetini eğitim veriseti ve test veriseti olmak üzere ikiye ayırıyorum
# 5- Standartizasyon : verisetini standartize ediyorum
# 6- model training and evaluation : tarin edip değerlendirme aşaması
# 7- hyperparameter tuning : başarımızı maximize edecek şekilde ayarlamak
# 8- model testing with real data : gerçek veriyle test ediyorum

import pandas as pd # data science library
import numpy as np # nümerik python kütüphanesi
import matplotlib.pyplot as plt 
import seaborn as sns # son iki kütüphane de görselleştirme kütüphanesi

from sklearn.preprocessing import StandardScaler #standartzasyonu gerçekleştirmek için standartscaler kullanıyoruz
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV #train ve test yapabilmek için trantest fonksiyonu, hyperparameter tuning için de GridSearchCV yöntemini kullanıyoruz
from sklearn.metrics import classification_report, confusion_matrix #ml modellerimizin sonuçlarını değerlendiricez

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

#2---------------------------------- Verisetini içeri aktarıp keşifsel veri analizi

#loading data
df = pd.read_csv("diabetes.csv")
df_name = df.columns #sürunları sonradan kullanabilelim diye df_name e atadık

df.info() #information fonksiyonu --> verilerimizin analizini gerçekleştirmek için // hatta kayıp verimiz var mı yok mu incelemek adına bakıyorum non-null hepsi harika
#sample sayısı ve kayıp veri var mı yok mu kontrol ettim
#sütun isimleri (büyük küçük fark, boşluk, ingilizce olmayan karakterler)
#veri tiplerini kontrol ettim age -> int evet ama string olsaydı sıkıntı olurdu

describe = df.describe() # describe fonk ise verisetimin içerisinde bulunan nümerik değerlerin temel istatistiksel özelliklerin analiz edildiği bir fonksiyon

#veri görselleştirme için seaborn kütüphanesini kullanıyorum
sns.pairplot(df, hue = "Outcome") #data frame'mi input olarak veriyorum, aynı zamanda bize bunu görselleştirmesiini ama bunun için data frame'in içindeki outcome'lara göre görselleştir(0 ve 1'e göre farklı renk ata).
plt.show()

#korelasyona dikkat çekiyorum burada,sayısal olarak korelasyonu değerlendiriyorum
def plot_correlation_heatmap(df):
    
    corr_matrix = df.corr()#data fram in içerisinde bulunan features'ların korelasyounu bulduk
    
    plt.figure(figsize =(10,8)) #görselleştirmek için kullandım
    sns.heatmap(corr_matrix, annot=True, cmap= "coolwarm", fmt = ".2f", linewidths=0.5) #seaborn kütüphanesinin heat map'i korelasyon sonuçlarını görselleştirmek için en iyisi
    plt.title("Correlation of Features")
    plt.show()

plot_correlation_heatmap(df)

#3------------------------------------ Outlier Detection (Aykırı Değer)

def detect_outliers_iqr(df): # iqr yöntemi:  sınırların dışında kalan veriyi outlier olarak adlandırmamda yaran yöntem
  
    outlier_indices = [] #outlier'ları saklayan bir liste oluşturdum ki daha sonra bu aykırı değerleriçıkarabilelim
    outliers_df = pd.DataFrame() #outlier'ları görselleştiremek için de DataFrame yaptım
    
    for col in df.select_dtypes(include=["float64", "int64"]).columns: #sadece nümerik değerleri içermesi için bir for döngüsü yazdım  
    
        
        Q1 = df[col].quantile(0.25) #first quartile 1.çeyrek
        Q3 = df[col].quantile(0.75) #third quartile
        
        IQR = Q3 - Q1 #interquartile range 
        
        #literatürde 1.5 ile çarpıldığı için ben de öyle devam ettim
        lower_bound = Q1 - 1.5 * IQR #mesela outlier sayımız çok çıktı o zaman sınırları genişletmemiz gerekiyor
        upper_bound = Q3 + 1.5 * IQR
        
        #bu verilerin dışında kalanlara outlier diyoruz yani şöyle       
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_indices.extend(outliers_in_col.index)
        
        outliers_df = pd.concat([outliers_df, outliers_in_col], axis = 0) #axis=0 satır satır ekleyebilmem için

    #remove duplicate indices #bir sample'da iki farklı outlier olabilir.
    outlier_indices = list(set(outlier_indices)) # Bunun için aynı sample'ı iki kere eklememek için birden fazla sample'ları çıkarmamız gerekiyor
    
    #remove duplicate rows in the outliers dataframe
    outliers_df = outliers_df.drop_duplicates() #dataframe'deki aynı satırları da çıkarmamiz gerekiyor
    #drop_duplicate özel bir fonksiyon birbiriyle aynı satırları çıkarmamı sağladı
    
    return outliers_df, outlier_indices
    
outliers_df, outlier_indices = detect_outliers_iqr(df)

# remove outliers from the dataframe
df_cleaned = df.drop(outlier_indices).reset_index(drop=True) #outlier indislerini verisetinden çıkarmak için 
#temizlenmiş veriseti = neyi drop ediyoruz? outlier'ımızın indislerini.

#4----------------------------------------- Train Test Split

#Öncelikle verisetinin içerisinde bulunan sample'ları ve outcome ları birbirinden ayıralım
#Temizlenmiş verisetinden outcome'ları çıkarırsam X değerimi elde etmiş olurum.

X = df_cleaned.drop(["Outcome"], axis = 1) #axis 1 outcome ı sütun olarak çıkar demek
y = df_cleaned["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42) # x ve y verisetini kullanarak x train ve test , y train ve test oluşturacak

#5------------------------------------------ Standartizasyon

scaler =StandardScaler()

X_train_scaled = scaler.fit_transform(X_train) #fit_transform işlemi gerçekleştirdim bu ne demek?
                                               #StandardScaler X train verisinden öğren sonra x train verisine uygula sonra x_train_scaled değerini elde et
X_test_scaled = scaler.transform(X_test)       # scaler zaten hazırdı o yzden sadece uyguladık yani transform yaptık


#6------------------------------------------ Model Training and Evaluation

""" Makine Öğrenmesi Modelleri

LogisticRegression 
DecisionTreeClassifier 
KNeighborsClassifier
GaussianNB
SVC
AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

"""
def getBasedModel(): #Modelleri oluşturup train edip test ile değerlendirebiliriz. Burada 8 tane modelimizi tanımladık
    basedModels = []
    basedModels.append(("LR",LogisticRegression()))
    basedModels.append(("DT",DecisionTreeClassifier()))
    basedModels.append(("KNN",KNeighborsClassifier()))
    basedModels.append(("NB",GaussianNB()))
    basedModels.append(("SVM",SVC()))
    basedModels.append(("AdaB",AdaBoostClassifier()))
    basedModels.append(("GBM",GradientBoostingClassifier()))
    basedModels.append(("RF",RandomForestClassifier()))

    return basedModels 

#Modelimizin eğitim aşamasına geldik. Modelimizin eğitimi sırasında KFoldValidation yöntemini kullanıyoruz

def baseModelsTraining(X_train, y_train, models):
    
    results = []
    names = []
    for name, model in models: #Bu modellerimizi KFold ve cross validation yöntemleriyle eğitimini gerçekleştirdim
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy") #çıkan sonuçları accuracy olarak depoladım
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy: {cv_results.mean()}, std: {cv_results.std()}") #ortalama accuracy değerlerine yani ortalama başarımlarına ve bu başarımların standart sapmalarını inceledim 
        
    return names, results

#hesapladığım accuracy ve standart sapma hesaplamalarını görselleştirmek için de box plot kullandım.
def plot_box(names, results):
    df = pd.DataFrame({names[i]: results[i] for i in range(len(names))})
    plt.figure(figsize = (12,8))
    sns.boxplot(data=df)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.show()
    
models = getBasedModel()
names, results = baseModelsTraining(X_train, y_train, models)
plot_box(names, results)

#7------------------------------------------------------------Hyperparameter Tuning
#Test sonuçlarına göre en başarılı modelimiz %78 ile LogictikRegression çıkmıştı. En başarısız modelimiz ise %67 başarım ortalamsı ile Decision Tree çıkmıştı
#Bu kısımda da Decision Tree'nin hyperparameter tuning ile başarımını arttırmaya çalışıyorum

param_grid = {
   "criterion": ["gini", "entropy"],
   "max_depth": [10, 20, 30, 40, 50],
   "min_samples_split": [2, 5, 10],
   "min_samples_leaf": [1,2,4]
   }

dt = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy")

#training
grid_search.fit(X_train, y_train)

print("En iyi parametreleri: ",grid_search.best_params_)

best_dt_model = grid_search.best_estimator_

y_pred = best_dt_model.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("classification_report")
print(classification_report(y_test, y_pred))

#8----------------------------------------------------------------Model Testing with Real Data

new_data = np.array([[6,149,72,35,0,34.6,0.627,51]])

new_prediction = best_dt_model.predict(new_data)






















