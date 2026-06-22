import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#load the saved file
df_new = joblib.load("data/Mall_Customers_updated.csv")
print(df_new.columns)

df_new = df_new.drop(columns = ['CustomerID'])
X = df_new

#train_test_split

X_train , X_test  = train_test_split(X ,  test_size=0.25 , random_state=42)

#Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#First find the best value of k 
wcss = []
for k in range(1 , 21):
    Kmeans = KMeans(n_clusters= k , init="k-means++")
    Kmeans.fit(X_train_scaled)
    wcss.append(Kmeans.inertia_)
print(wcss)

#Now apply knee locator 
kn = KneeLocator(range(1 , 21) , wcss , curve="convex" , direction="decreasing")
print("kn - value : " , kn.knee)

#now apply k means
Kmeans2 = KMeans(n_clusters = kn.knee , init="k-means++" , random_state=42 , n_init=10)

y_train = Kmeans2.fit_predict(X_train_scaled)
y_test = Kmeans2.predict(X_test_scaled)

#Now apply random forest
model = RandomForestClassifier(
    n_estimators=100 ,
    random_state=42 ,
    max_depth=3
)
rand_trained_data = model.fit(X_train_scaled , y_train)
print(f"Validation score : {rand_trained_data.score(X_test_scaled , y_test) * 100:.2f} %")

#Now print score
#Analyze the clusters 
analysis = X_train.copy()
analysis['Cluster'] = y_train
print(analysis.groupby('Cluster').mean())
