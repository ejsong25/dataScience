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

# plot 한글 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 정규화된 전세 데이터셋 로드
jeonse_data = pd.read_csv("jeonse_dataset_normalized.csv")

""" regression (특정 조건(도로상태, 면적, 계약기간, 방 개수, 건물연식)에 따른 전세 보증금 예측) """

# independent variable와 target variable 설정
X = jeonse_data.drop("deposit", axis=1)  # independent variables
y = jeonse_data["deposit"]  # target variable (continuous)

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
print(f"Mean_Absolute_Error: {mae}, R2_Score: {r2}\n")

# 예측 결과 시각화
plt.scatter(y_test, y_pred, s=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r-", lw=1)
plt.xlabel("Actual Deposit")
plt.ylabel("Predicted Deposit")
plt.title("Linear Regression Plot")
plt.show()


""" classification (전세 보증금을 특정 기준에 따라 분류) """

# 전세 보증금의 mean, std 계산
deposit_mean = jeonse_data["deposit"].mean()
deposit_std = jeonse_data["deposit"].std()

# 수치형 레이블로 분류 기준 설정
criterion_labels = [
    0,  # 매우 저렴
    1,  # 저렴
    2,  # 보통
    3,  # 비쌈
    4,  # 매우 비쌈
]

# 전세 보증금을 분류 기준에 따라 범주화
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

# independent variable과 target variable 설정
X = jeonse_data.drop(["deposit", "deposit_category"], axis=1)  # independent variables
y = jeonse_data["deposit_category"]  # target variable

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
# print(cm)

# Accuracy, Precision, Recall, F1 Score 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1_ = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_}")

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
plt.title("Jeonse Deposit Confusion Matrix")
plt.show()
