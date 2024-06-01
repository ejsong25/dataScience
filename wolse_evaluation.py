import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import (train_test_split,
                                     cross_val_score,
                                     KFold,
                                     RepeatedKFold,
                                     StratifiedKFold,
                                     RepeatedStratifiedKFold
                                     )
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Preventing Korean crush in plots
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# Load normalized wolse dataset
wolse_data = pd.read_csv("wolse_dataset_normalized.csv")

""" Regression (Predicting Monthly Rent Bill Based on Specific Conditions 
(Road Condition, Contract Area, Contract Period, Number of Rooms, Building Age) """

# Setting Independent variable and Target variable
X = wolse_data.drop("monthly_rent_bill", axis=1)  # independent variables
y = wolse_data["monthly_rent_bill"]  # target variable (continuous)

# Spliting data into train_set and test_set
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

# 10-Fold Cross Validation 
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Repeated 10-Fold Cross Validation
repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

# CVS Calculation
cvs = cross_val_score(model, X, y, cv=kfold)
repeated_cvs = cross_val_score(model, X, y, cv=repeated_kfold, scoring='r2')

# Wolse dataset Linear Regression Model Evaluation
print("<Monthly-Rent dataset Linear Regression Model Evaluation>")
print(f"1.Mean_Absolute_Error: {mae}, R2_Score: {r2}\n")

# CVS values and CVS Average Value Comparison
print("CVS Value")
print("(1)KFold CVS\n", cvs)
print()
print("(2)Repeated KFold CVS\n", repeated_cvs)
print()
print("2.CVS Mean Comparison")
print(f"CVS Mean: {cvs.mean()}")
print(f"Repeated KFold CVS Mean: {repeated_cvs.mean()}\n\n")

# Prediction Visualization
plt.scatter(y_test, y_pred, s=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r-", lw=1)
plt.xlabel("Actual Monthly Rent Bill")
plt.ylabel("Predicted Monthly Rent Bill")
plt.title("Linear Regression Plot")
#plt.show()


""" classification (Classify monthly rent according to specific criteria) """

# Calculating monthly rent bill's mean, std
monthly_rent_mean = wolse_data["monthly_rent_bill"].mean()
monthly_rent_std = wolse_data["monthly_rent_bill"].std()

# Setting Classification Criteria for Numerical Labels
criterion_labels = [
    0,  # very cheap
    1,  # cheap
    2,  # appropriate
    3,  # expensive
    4,  # very expensive
]

# Categorizing Monthly rent bill According to Classification Criteria
"""
    very cheap: -inf ~ (monthly_rent_mean - 1.5 * monthly_rent_std)
    cheap: (monthly_rent_mean - 1.5 * monthly_rent_std) ~ (monthly_rent_mean - 0.5 * monthly_rent_std))
    appropriate: (monthly_rent_mean - 0.5 * monthly_rent_std) ~ (monthly_rent_mean + 0.5 * monthly_rent_std))
    expensive: (monthly_rent_mean + 0.5 * monthly_rent_std) ~ (monthly_rent_mean + 1.5 * monthly_rent_std))
    very expensive: (monthly_rent_mean + 1.5 * monthly_rent_std) ~ inf
"""
wolse_data["monthly_rent_bill_category"] = pd.cut(
    wolse_data["monthly_rent_bill"],
    bins=[
        -float("inf"),
        monthly_rent_mean - 1.5 * monthly_rent_std,
        monthly_rent_mean - 0.5 * monthly_rent_std,
        monthly_rent_mean + 0.5 * monthly_rent_std,
        monthly_rent_mean + 1.5 * monthly_rent_std,
        float("inf"),
    ],
    labels=criterion_labels,
    right=False,
)

# Setting independent variable and target variable
X = wolse_data.drop(
    ["monthly_rent_bill", "monthly_rent_bill_category"], axis=1
)  # independent variables
y = wolse_data["monthly_rent_bill_category"]  # target variable

# Splitting data into train_set and test_set
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

# Performing prediction on the test_set
y_pred = model.predict(X_test)

# Confusion Matrix 생성 (Including all labels)
labels = np.arange(len(criterion_labels))
cm = confusion_matrix(y_test, y_pred, labels=labels).T
#print(cm)

# Accuracy, Precision, Recall, F1 Score Calculation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1_ = f1_score(y_test, y_pred, average="weighted")

# 10-Fold Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Stratified 10-Fold Cross Validation
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Repeated 10-Fold Cross Validation
repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

# Repeated Stratified 10-Fold Cross Validation
repeated_stratified_kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

# CVS Calculation
cvs = cross_val_score(model, X, y, cv=kfold)
stratified_cvs = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
repeated_cvs = cross_val_score(model, X, y, cv=repeated_kfold, scoring='accuracy')
repeated_stratified_cvs = cross_val_score(model, X, y, cv=repeated_stratified_kfold, scoring='accuracy')

# CVS Values
print("CVS Value")
print("(1)KFold CVS\n", cvs)
print()
print("(2)Stratified KFold CVS\n", stratified_cvs)
print()
print("(3)Repeated KFold CVS\n", repeated_cvs)
print()
print("(4)Repeated Stratified KFold CVS\n", repeated_stratified_cvs)
print()

# Wolse dataset Classification Model Evaluation
print("<Monthly-Rent dataset Classification Model Evaluation>")

# Show Accuracy, Precision, Recall, F1 Score
print("1.Accuracy, Precision, Recall, F1 Score") 
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_}\n")

# Comparison of Average Values from Each KFold Method CSV
print("2.KFold CVS Mean Comparison")
print(f"KFold CVS Mean: {cvs.mean()}\n"
      f"Stratified KFold CVS Mean: {stratified_cvs.mean()}\n"
      f"Repeated KFold CVS Mean: {repeated_cvs.mean()}\n"
      f"Repeated Stratified KFold CVS Mean: {repeated_stratified_cvs.mean()}\n"
      )

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
plt.title("Wolse Monthly Rent Bill Confusion Matrix")
#plt.show()
