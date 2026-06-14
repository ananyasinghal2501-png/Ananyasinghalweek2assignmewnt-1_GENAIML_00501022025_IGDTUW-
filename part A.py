import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#Part A: Dataset Understanding 
#Q1
df = pd.read_csv(r"C:\Users\drman\Downloads\Dataset 2 (1).csv")
print(df.head())
#Q2
print('Shape:', df.shape)
#Q3
print('\nColumns:')
print(df.columns)
#Q4
numerical_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
categorical_features = list(df.select_dtypes(include=['object', 'string']).columns)

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)
#Q5
print("\nMissing Values:")
print(df.isnull().sum())
