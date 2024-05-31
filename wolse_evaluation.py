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

# plot 한글 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 정규화된 월세 데이터셋 로드
wolse_data = pd.read_csv("wolse_dataset_normalized.csv")

""" regression (특정 조건(도로상태, 면적, 계약기간, 방 개수, 건물연식)에 따른 월세 예측) """

# independent variable와 target variable 설정
X = wolse_data.drop("monthly_rent_bill", axis=1)  # independent variables
y = wolse_data["monthly_rent_bill"]  # target variable (continuous)

# 데이터를 train_set와 test_set로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    shuffle=True,
    random_state=np.random.seed(),
)

# Linear Regression 모델 생성 및 훈련
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 세트에 대한 예측 수행
y_pred = model.predict(X_test)

# RMSE, R2 Score 계산
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 10-Fold Cross Validation 
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Repeated 10-Fold Cross Validation
repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

# CVS 계산
cvs = cross_val_score(model, X, y, cv=kfold)
repeated_cvs = cross_val_score(model, X, y, cv=repeated_kfold, scoring='r2')

# 전세 데이터셋 Linear Regression Model Evaluation
print("<Monthly-Rent dataset Linear Regression Model Evaluation>")
print(f"1.Mean_Absolute_Error: {mae}, R2_Score: {r2}\n")

# CVS 값과 CVS 평균값 비교
print("CVS Value")
print("(1)KFold CVS\n", cvs)
print()
print("(2)Repeated KFold CVS\n", repeated_cvs)
print()
print("2.CVS Mean Comparison")
print(f"CVS Mean: {cvs.mean()}")
print(f"Repeated KFold CVS Mean: {repeated_cvs.mean()}\n\n")

# 예측 결과 시각화
plt.scatter(y_test, y_pred, s=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r-", lw=1)
plt.xlabel("Actual Monthly Rent Bill")
plt.ylabel("Predicted Monthly Rent Bill")
plt.title("Linear Regression Plot")
#plt.show()


""" classification (월세를 특정 기준에 따라 분류) """

# 월세의 mean, std 계산
monthly_rent_mean = wolse_data["monthly_rent_bill"].mean()
monthly_rent_std = wolse_data["monthly_rent_bill"].std()

# 수치형 레이블로 분류 기준 설정
criterion_labels = [
    0,  # 매우 저렴
    1,  # 저렴
    2,  # 보통
    3,  # 비쌈
    4,  # 매우 비쌈
]

# 월세를 분류 기준에 따라 범주화
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

# independent variable과 target variable 설정
X = wolse_data.drop(
    ["monthly_rent_bill", "monthly_rent_bill_category"], axis=1
)  # independent variables
y = wolse_data["monthly_rent_bill_category"]  # target variable

# 데이터를 train_set와 test_set로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    shuffle=True,
    stratify=y,
    random_state=40,
)

# Logistic Regression 모델 생성 및 훈련
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# test_set에 대한 예측 수행
y_pred = model.predict(X_test)

# confusion matrix 생성 (모든 레이블 포함)
labels = np.arange(len(criterion_labels))
cm = confusion_matrix(y_test, y_pred, labels=labels).T
#print(cm)

# Accuracy, Precision, Recall, F1 Score 계산
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

# CVS 계산
cvs = cross_val_score(model, X, y, cv=kfold)
stratified_cvs = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
repeated_cvs = cross_val_score(model, X, y, cv=repeated_kfold, scoring='accuracy')
repeated_stratified_cvs = cross_val_score(model, X, y, cv=repeated_stratified_kfold, scoring='accuracy')

# CVS 값들
print("CVS Value")
print("(1)KFold CVS\n", cvs)
print()
print("(2)Stratified KFold CVS\n", stratified_cvs)
print()
print("(3)Repeated KFold CVS\n", repeated_cvs)
print()
print("(4)Repeated Stratified KFold CVS\n", repeated_stratified_cvs)
print()

# 전세 데이터셋 Classification Model Evaluation
print("<Monthly-Rent dataset Classification Model Evaluation>")

# Show Accuracy, Precision, Recall, F1 Score
print("1.Accuracy, Precision, Recall, F1 Score") 
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_}\n")

# 각 KFold 메서드 CSV 평균값 비교
print("2.KFold CVS Mean Comparison")
print(f"KFold CVS Mean: {cvs.mean()}\n"
      f"Stratified KFold CVS Mean: {stratified_cvs.mean()}\n"
      f"Repeated KFold CVS Mean: {repeated_cvs.mean()}\n"
      f"Repeated Stratified KFold CVS Mean: {repeated_stratified_cvs.mean()}\n"
      )

# Confusion Matrix 시각화
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