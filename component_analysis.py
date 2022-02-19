# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 14:44:16 2021

@author: ceren
"""


#%% Kütüphanelerin import edilmesi

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Datasetinin düzenlenmesi ve incelenmesi

df = pd.read_csv("proje_dataset.csv", sep = ";")
df.info()

df.drop(["zaman","cinsiyet", "boy ve kilo",], axis = 1, inplace = True)
df.info()

#görselleştirme
sns.set_style("darkgrid")
sns.countplot(df["diyabet"])

print("Diyabet olan ve olmayan kişi sayısı:\n", df.diyabet.value_counts())

# String değerleri sayısal değere çevirme
df["diyabet"] = [1 if i.strip()== "Evet" else 0 for i in df.diyabet]


describe = df.describe()


#Korelasyon
corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Özellikler arasındaki korelasyon:")
plt.show() #BMI ve glikoz ile diyabet arasında yüksek ihtimalle bir ilişki var.


# Outcome ve features
y = df.diyabet
x = df.drop(["diyabet"], axis =1)


#%% Sklearn Kütüphaneleri ile Principal Component Analizi

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


pca = PCA(n_components = 5)
pca.fit(x_scaled)

# pc1, pc2, pc2, pc3,pc4 ve pc5 
x_new = pca.transform(x_scaled)

pca_df = pd.DataFrame(x_new, columns=["PC1","PC2","PC3","PC4","PC5"])
pca_df["diyabet"] = y

# yeni oluşan PC değerlerinin dataframe'e eklenmesi
df["PC1"] = x_new[:,0]
df["PC2"] = x_new[:,1]
df["PC3"] = x_new[:,2]
df["PC4"] = x_new[:,3]
df["PC5"] = x_new[:,4]


#görselleştirme (PC1 ve diğer PC'ler arasındaki ilişki)

sns.scatterplot(x = "PC1", y ="PC2", hue ="diyabet", data= pca_df)
plt.title("Principal Component Analysis: PC1 and PC2")

sns.scatterplot(x = "PC1", y ="PC3", hue ="diyabet", data= pca_df)
plt.title("Principal Component Analysis: PC1 and PC3")

sns.scatterplot(x = "PC1", y ="PC4", hue ="diyabet", data= pca_df)
plt.title("Principal Component Analysis: PC1 and PC4")

sns.scatterplot(x = "PC1", y ="PC5", hue ="diyabet", data= pca_df)
plt.title("Principal Component Analysis: PC1 and PC5")


#Varyans ve toplamları

print("Variance ratio:" , pca.explained_variance_ratio_)
variance_ratio = pca.explained_variance_ratio_
print("Sum:", sum(variance_ratio))

"""

# 2 COMPONENTLİ PCA ANALİZİ

pca = PCA(n_components = 2 ) #2 temel bileşene düşürüldü.
pca.fit(x_scaled)

# pc1 ve pc2 
x_new = pca.transform(x_scaled)

pca_df = pd.DataFrame(x_new, columns=["PC1","PC2"])
pca_df["diyabet"] = y

# yeni oluşan p1 ve p2 değerlerinin dataframe'e eklenmesi
df["PC1"] = x_new[:,0]
df["PC2"] = x_new[:,1]

#görselleştirme
sns.scatterplot(x = "PC1", y ="PC2", hue ="diyabet", data= pca_df)
plt.title("Principal Component Analysis: PC1 and PC2")


#Varyans ve toplamları

print("Variance ratio:" , pca.explained_variance_ratio_)
variance_ratio = pca.explained_variance_ratio_
print("Sum:", sum(variance_ratio))
"""


#%% Principal Component Analizi -2

# Kovaryans Matrisi
features = x_scaled.T
covariance_matrix = np.cov(features)
print("Kovaryans matrisi:", covariance_matrix)

# öz vektör ve öz değerlerinin bulunması
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

print("Eigen vectors\n", eigen_vectors)
print("\n Eigen values\n", eigen_values)

# Varyansların bulunması
varyans1 = eigen_values[0]/ sum(eigen_values)
print("Varyans1:\n", varyans1)  #0.422978...

varyans2 = eigen_values[1]/ sum(eigen_values)
print("Varyans2:", varyans2)   #0.238243...

varyans3 = eigen_values[2]/ sum(eigen_values)
print("Varyans3:", varyans3)   #0.0758748...

varyans4 = eigen_values[3]/ sum(eigen_values)
print("Varyans4:", varyans4)   #0.104859...

varyans5 = eigen_values[4]/ sum(eigen_values)
print("Varyans5:", varyans5)   #0.158043...


toplam_varyans = varyans1 + varyans2 + varyans3 + varyans4 + varyans5
print("Toplam varyans:", toplam_varyans) #1.0

#PC değerlerinin bulunması
PC1 = x_scaled.dot(eigen_vectors.T[0])
PC2 = x_scaled.dot(eigen_vectors.T[1])
PC3 = x_scaled.dot(eigen_vectors.T[2])
PC4 = x_scaled.dot(eigen_vectors.T[3])
PC5 = x_scaled.dot(eigen_vectors.T[4])


#%% Train-Test Split

x_pca = pca_df.drop(["diyabet"], axis =1 )
y_pca = pca_df.diyabet


from sklearn.model_selection import train_test_split
test_size = 0.21
x_train, x_test, y_train, y_test = train_test_split(x_pca, y_pca, test_size= test_size, random_state = 42)

print("x_train", x_train.shape)
print("x_test", x_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)


#%% Multiple Linear Regression Model

# y = b0 + b1x1 + b2x2 + ...

from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)


# intercept ve coefficientların bulunması
print("b0:", linear_reg.intercept_)
print("b1, b2, b3, b4 ve b5:", linear_reg.coef_)

y_predicted = linear_reg.predict(x_test).reshape(-1,1)
print("tahmin edilen değerler:", y_predicted)

y_predicted = [1 if i>0.5 else 0 for i in y_predicted]

#test değerlerine göre tahmin
print("tahmin edilen değerler:", y_predicted)
print("gercek degerler:", y_test)


"""
plt.plot(x_test, y_test, color="red", label="x test ve y test arasındaki ilişki")
plt.legend()
plt.show()
"""


#%% Results

# mean squarred error
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_predicted)
print("Mean Squarred Error:\n", mse)    #0.0555... (random state=42), 0.1111.. (rs=1)

# confusion metrics and statistics

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", cm)

#accuracy score
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test,y_predicted)
print("Accuracy Score:", acc_score)






























































