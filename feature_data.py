import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

#Load the dataset
df = pd.read_csv("data/Mall_Customers.csv" )
print(df.head(10))
print("-" * 100)
print("Shape : " , df.shape)
print("-" * 100)


#Check null values
print(df.isnull().sum())
print("-" * 100)

#Columns
print(df.columns)
print("-" * 100)

#Info
print(df.info())


#Doing one hot coding for gender column 
encoder = OneHotEncoder(sparse_output=False , drop='first' )
encoded = encoder.fit_transform(df[['Gender']])
encoded_df = pd.DataFrame(encoded , columns=encoder.get_feature_names_out(['Gender']))
df = pd.concat([df , encoded_df ] , axis=1 ).drop(columns = ['Gender'] )



