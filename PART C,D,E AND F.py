import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\drman\Downloads\Dataset 2 (1).csv")

#Q11 convert categorical features into numerical form
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["SubscriptionType"] = le.fit_transform(df["SubscriptionType"])
df["FavoriteGenre"] = le.fit_transform(df["FavoriteGenre"])
df["SubscriptionRenewed"] = le.fit_transform(df["SubscriptionRenewed"])
print(df.head())

#Q12 define features and target
X = df.drop("SubscriptionRenewed", axis=1)
y = df["SubscriptionRenewed"]
print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

#Q13 split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

#PART D DECISION TREE 
#Q14 train Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree Model Trained Successfully")

#Q15predict and Calculate Accuracy
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Q16 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

#PART E 

# Q17. Train KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
# Make Predictions
y_pred_knn = knn_model.predict(X_test)\
# Calculate Accuracy
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", knn_accuracy)

# Q18. Compare Accuracies

print("Decision Tree Accuracy:", accuracy)
print("KNN Accuracy:", knn_accuracy)

if knn_accuracy > accuracy:
    print("KNN performs better than Decision Tree.")
elif knn_accuracy < accuracy:
    print("Decision Tree performs better than KNN.")
else:
    print("Both models have the same accuracy.")

#PART F LinearRegression
# Q19. Define Features and Target

X = df.drop("MonthlySpend", axis=1)
y = df["MonthlySpend"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
# Train Linear Regression Model

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("LR Model Trained Successfully")

#Q20
new_user = [[2001, 25, 0, 1, 15, 3, 2, 20, 1]]
predicted_spend = lr_model.predict(new_user)
print("Predicted Monthly Spending:", predicted_spend[0])
