import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Preventing Koerean crush in plots
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# Load normalized jeonse dataset
jeonse_data = pd.read_csv("jeonse_dataset_normalized.csv")

""" Regression (Predicting Monthly Rent Bill Based on Specific Conditions 
(Road Condition, Contract Area, Contract Period, Number of Rooms, Building Age) """

# Setting Independent variable and Target variable
X = jeonse_data.drop("deposit", axis=1)  # independent variables
y = jeonse_data["deposit"]  # target variable (continuous)

# Splitting data into train_set and test_set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    shuffle=True,
    random_state=np.random.seed(),
)

# Linear Regression Modeling and Training
model = LinearRegression()
model.fit(X_train, y_train)

# Performing Predictions on the Test Set
y_pred = model.predict(X_test)

# RMSE, R2 Score Calculation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean_Absolute_Error: {mae}, R2_Score: {r2}\n")

# Prediction Visualization
plt.scatter(y_test, y_pred, s=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r-", lw=1)
plt.xlabel("Actual Deposit")
plt.ylabel("Predicted Deposit")
plt.title("Linear Regression Plot")
#plt.show()


""" classification (Classify jeonse deposit according to specific criteria) """

# Jeonse Deposit mean, std calculation
deposit_mean = jeonse_data["deposit"].mean()
deposit_std = jeonse_data["deposit"].std()

# Setting Classification Criteria for Numerical Labels
criterion_labels = [
    0,  # very cheap
    1,  # cheap
    2,  # appropriate
    3,  # expensive
    4,  # very expensive
]

# Categorizing Jeonse Deposit According to Classification Criteria
'''
    very cheap: -inf ~ (deposit_mean - 1.5 * deposit_std)
    cheap: (deposit_mean - 1.5 * deposit_std) ~ (deposit_mean - 0.5 * deposit_std)
    appropriate: (deposit_mean - 0.5 * deposit_std) ~ (deposit_mean + 0.5 * deposit_std)
    expensive: (deposit_mean + 0.5 * deposit_std) ~ (deposit_mean + 1.5 * deposit_std)
    very expensive: (deposit_mean + 1.5 * deposit_std) ~ inf
'''
jeonse_data["deposit_category"] = pd.cut(
    jeonse_data["deposit"],
    bins=[
        -float("inf"),
        deposit_mean - 1.5 * deposit_std,
        deposit_mean - 0.5 * deposit_std,
        deposit_mean + 0.5 * deposit_std,
        deposit_mean + 1.5 * deposit_std,
        float("inf"),
    ],
    labels=criterion_labels,
    right=False,
)

# Setting independent variable and target variable
X = jeonse_data.drop(["deposit", "deposit_category"], axis=1)  # independent variables
y = jeonse_data["deposit_category"]  # target variable

# Splitting dataset int train_set and  test_set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    shuffle=True,
    stratify=y,
    random_state=40,
)

# Logistic Regression Modeling and Training
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Performing Prediction on test_set
y_pred = model.predict(X_test)

# Confusion Matrix (Including all labels)
labels = np.arange(len(criterion_labels))
cm = confusion_matrix(y_test, y_pred, labels=labels).T
# print(cm)

# Accuracy, Precision, Recall, F1 Score Calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1_ = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_}")

# Confusion Matrix Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=["very cheap", "cheap", "appropriate", "expensive", "very expensive"],
    yticklabels=["very cheap", "cheap", "appropriate", "expensive", "very expensive"],
)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Jeonse Deposit Confusion Matrix")
#plt.show()

class_cnt =jeonse_data["deposit_category"].value_counts()
total_samples = len(jeonse_data)
class_ratio = class_cnt / total_samples
print(class_ratio)