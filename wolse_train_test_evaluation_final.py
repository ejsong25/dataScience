import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     cross_val_score,
                                     KFold,
                                     RepeatedKFold,
                                     StratifiedKFold,
                                     RepeatedStratifiedKFold)

from sklearn.metrics import (
    classification_report,
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

# Load normalized wolse datasets (onehot encoding, district score method)
wolse_data_score = pd.read_csv(
    "dataScience/wolse_dataset_normalized_score.csv")
wolse_data_onehot = pd.read_csv(
    "dataScience/wolse_dataset_normalized_onehot.csv")

# Setting different test sizes
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]


""" Regression (Predicting Monthly Rent Bill Based on Specific Conditions 
(Road Condition, Contract Area, Contract Period, Number of Rooms, Building Age) """

""" score method를 사용한 데이터셋 """

# Setting Independent variable and Target variable
X = wolse_data_score.drop("monthly_rent_bill", axis=1)  # independent variables
y = wolse_data_score["monthly_rent_bill"]  # target variable (continuous)

print("<wolse_dataset_normalized_score - Monthly-Rent dataset Linear Regression Model Evaluation>")

for test_size in test_sizes:
    # Split dataset into train_set and test_set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=42,
    )

    # Linear Regression Modeling and Training
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Perform prediction on test set
    y_pred = linear_model.predict(X_test)

    # RMSE, R2 Score Calculation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 10-Fold Cross Validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Repeated 10-Fold Cross Validation
    repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    # CVS Calculation
    cvs = cross_val_score(linear_model, X, y, cv=kfold)
    repeated_cvs = cross_val_score(
        linear_model, X, y, cv=repeated_kfold, scoring='r2')

    # Bootstrap Method
    n_iterations = 1000
    n_size = int(len(X) * 0.5)
    bootstrap_scores = []

    for _ in range(n_iterations):
        X_bs, y_bs = resample(X, y, n_samples=n_size, random_state=42)
        linear_model.fit(X_bs, y_bs)
        score = linear_model.score(X_test, y_test)
        bootstrap_scores.append(score)

    bootstrap_mean = np.mean(bootstrap_scores)

    # Display Result
    print(f"\nTest Size: {test_size}")
    print(f"1. Mean_Absolute_Error: {mae}, R2_Score: {r2}\n")

    # CVS values, CVS Average value comparison, Bootstrap, and Bootstrap Mean
    print("CVS Value")
    print("(1) KFold CVS\n", cvs)
    print()
    print("(2) Repeated KFold CVS\n", repeated_cvs)
    print()
    # Print the first 10 scores for brevity
    print("(3) Bootstrap Method\n", bootstrap_scores[:10], "...")
    print()
    print("2. CVS Mean Comparison")
    print(f"KFold CVS Mean: {cvs.mean()}")
    print(f"Repeated KFold CVS Mean: {repeated_cvs.mean()}")
    print(f"Bootstrap Mean: {bootstrap_mean}\n\n")

    # Prediction Visualization
    plt.scatter(y_test, y_pred, s=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r-", lw=1)
    plt.xlabel("Actual Monthly Rent Bill")
    plt.ylabel("Predicted Monthly Rent Bill")
    plt.title("Linear Regression Plot(score)")
    plt.show()


""" onehot encoding method를 사용한 데이터셋 """

# Setting Independent variable and Target variable
X = wolse_data_onehot.drop(
    "monthly_rent_bill", axis=1)  # independent variables
y = wolse_data_onehot["monthly_rent_bill"]  # target variable (continuous)

print("<wolse_dataset_normalized_one_hot - Monthly-Rent dataset Linear Regression Model Evaluation>")

for test_size in test_sizes:
    # Split dataset into train_set and test_set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=42,
    )

    # Linear Regression Modeling and Training
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Perform prediction on test set
    y_pred = linear_model.predict(X_test)

    # RMSE, R2 Score Calculation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 10-Fold Cross Validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # Repeated 10-Fold Cross Validation
    repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    # CVS Calculation
    cvs = cross_val_score(linear_model, X, y, cv=kfold)
    repeated_cvs = cross_val_score(
        linear_model, X, y, cv=repeated_kfold, scoring='r2')

    # Bootstrap Method
    n_iterations = 1000
    n_size = int(len(X) * 0.5)
    bootstrap_scores = []

    for _ in range(n_iterations):
        X_bs, y_bs = resample(X, y, n_samples=n_size, random_state=42)
        linear_model.fit(X_bs, y_bs)
        score = linear_model.score(X_test, y_test)
        bootstrap_scores.append(score)

    bootstrap_mean = np.mean(bootstrap_scores)

    # Display Result
    print(f"\nTest Size: {test_size}")
    print(f"1. Mean_Absolute_Error: {mae}, R2_Score: {r2}\n")

    # CVS values, CVS Average value comparison, Bootstrap, and Bootstrap Mean
    print("CVS Value")
    print("(1) KFold CVS\n", cvs)
    print()
    print("(2) Repeated KFold CVS\n", repeated_cvs)
    print()
    # Print the first 10 scores for brevity
    print("(3) Bootstrap Method\n", bootstrap_scores[:10], "...")
    print()
    print("2. CVS Mean Comparison")
    print(f"KFold CVS Mean: {cvs.mean()}")
    print(f"Repeated KFold CVS Mean: {repeated_cvs.mean()}")
    print(f"Bootstrap Mean: {bootstrap_mean}\n\n")

    # Prediction Visualization
    plt.scatter(y_test, y_pred, s=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r-", lw=1)
    plt.xlabel("Actual Monthly Rent Bill")
    plt.ylabel("Predicted Monthly Rent Bill")
    plt.title("Linear Regression Plot(one-hot encoding)")
    plt.show()

""" 
classification ((Classify monthly rent according to specific criteria) 
"""
""" Dataset that used score method """

# Calculating monthly rent bill's mean, std
monthly_rent_mean = wolse_data_score["monthly_rent_bill"].mean()
monthly_rent_std = wolse_data_score["monthly_rent_bill"].std()

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
wolse_data_score["monthly_rent_bill_category"] = pd.cut(
    wolse_data_score["monthly_rent_bill"],
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
X = wolse_data_score.drop(
    ["monthly_rent_bill", "monthly_rent_bill_category"], axis=1
)  # independent variables
y = wolse_data_score["monthly_rent_bill_category"]  # target variable

print("<wolse_dataset_normalized_scored - Monthly-Rent dataset DecisionTree Model Evaluation>")

for test_size in test_sizes:
    # Split dataset into train_set and test_set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=42,
    )

    # Decision Tree Modeling and Training
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    # Performing prediction on the test_set
    y_pred = dt_model.predict(X_test)

    # RMSE, R2 Score Calculation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Accuracy, Precision, Recall, F1 Score Calculation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1_ = f1_score(y_test, y_pred, average="weighted")

    # 10-Fold Cross Validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    stratified_kfold = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=42)
    repeated_stratified_kfold = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=10, random_state=42)

    # CVS Calculation
    cvs = cross_val_score(dt_model, X, y, cv=kfold)
    repeated_cvs = cross_val_score(
        dt_model, X, y, cv=repeated_kfold, scoring='accuracy')
    stratified_cvs = cross_val_score(
        dt_model, X, y, cv=stratified_kfold, scoring='accuracy')
    repeated_stratified_cvs = cross_val_score(
        dt_model, X, y, cv=repeated_stratified_kfold, scoring='accuracy')

    # Bootstrap Method
    n_iterations = 1000
    n_size = int(len(X) * 0.5)
    bootstrap_scores = []

    for _ in range(n_iterations):
        X_bs, y_bs = resample(X, y, n_samples=n_size, random_state=42)
        dt_model.fit(X_bs, y_bs)
        score = dt_model.score(X_test, y_test)
        bootstrap_scores.append(score)

    bootstrap_mean = np.mean(bootstrap_scores)

    # Display Result
    print(f"\nTest Size: {test_size}")
    print(f"1. Mean_Absolute_Error: {mae}, R2_Score: {r2}\n")

    # CVS values, CVS Average value comparison, Bootstrap, and Bootstrap Mean
    print("CVS Value")
    print("(1)KFold CVS\n", cvs)
    print()
    print("(2)Repeated KFold CVS\n", repeated_cvs)
    print()
    print("(3)Stratified KFold CVS\n", stratified_cvs)
    print()
    print("(4)Repeated Stratified KFold CVS\n", repeated_stratified_cvs)
    print()
    # Print the first 10 scores for brevity
    print("(5) Bootstrap Method\n", bootstrap_scores[:10], "...")
    print()
    print("2. CVS Mean Comparison")
    print(f"KFold CVS Mean: {cvs.mean()}")
    print(f"Repeated KFold CVS Mean: {repeated_cvs.mean()}")
    print(f"Stratified KFold CVS Mean: {stratified_cvs.mean()}")
    print(
        f"Repeated Stratified KFold CVS Mean: {repeated_stratified_cvs.mean()}")
    print(f"Bootstrap Mean: {bootstrap_mean}")

    # Show Accuracy, Precision, Recall, F1 Score
    print(f"Decision_tree_Accuracy: {accuracy}")
    print(f"Decision_tree_Precision: {precision}")
    print(f"Decision_tree_Recall: {recall}")
    print(f"Decision_tree_F1 Score: {f1_}")
    print(classification_report(y_test, y_pred))

    # confusion matrix (including all labels)
    labels = np.arange(len(criterion_labels))
    cm = confusion_matrix(y_test, y_pred, labels=labels).T
    print(f"cm(score method)\n{cm}\n")

    # Confusion Matrix Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["very cheap", "cheap",
                     "appropriate", "expensive", "very expensive"],
        yticklabels=["very cheap", "cheap",
                     "appropriate", "expensive", "very expensive"],
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Wolse Deposit Confusion Matrix (score method)")
    plt.show()

    # # 훈련된 결정 트리 모델의 트리 구조 시각화
    # plt.figure(figsize=(20, 10))  # 그래프의 크기 조절
    # plot_tree(
    #     dt_model,
    #     filled=True,
    #     rounded=True,
    #     class_names=["very cheap", "cheap", "appropriate", "expensive", "very expensive"],
    #     feature_names=X_train.columns,
    # )
    # plt.title("Decision Tree Visualization")
    # plt.show()

    # 하이퍼파라미터 그리드 설정
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 8, 10, 12, 15],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "criterion": ["gini", "entropy"],
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(
        estimator=dt_model,
        param_grid=param_grid,
        cv=5,
        verbose=1,
        scoring="accuracy",
    )

    # 그리드 탐색 실행
    grid_search.fit(X_train, y_train)

    # 최적의 하이퍼파라미터와 그 성능 출력
    print("Best parameters:", grid_search.best_params_)
    print(
        "Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # 최적의 모델로 예측 및 평가
    best_dt = grid_search.best_estimator_
    y_pred = best_dt.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 최적화된 모델의 confusion matrix 시각화
    cm_optimized = confusion_matrix(y_test, y_pred, labels=labels).T
    print(f"cm_optimized(score method)\n{cm_optimized}\n")
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cm_optimized,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["very cheap", "cheap",
                     "appropriate", "expensive", "very expensive"],
        yticklabels=["very cheap", "cheap",
                     "appropriate", "expensive", "very expensive"],
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Optimized Decision Tree Confusion Matrix")
    plt.show()


############################################################################################################
""" Dataset using onehot encoding method """

# Calculating monthly rent bill's mean, std
monthly_rent_mean = wolse_data_onehot["monthly_rent_bill"].mean()
monthly_rent_std = wolse_data_onehot["monthly_rent_bill"].std()

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
wolse_data_onehot["monthly_rent_bill_category"] = pd.cut(
    wolse_data_onehot["monthly_rent_bill"],
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
X = wolse_data_onehot.drop(
    ["monthly_rent_bill", "monthly_rent_bill_category"], axis=1
)  # independent variables
y = wolse_data_onehot["monthly_rent_bill_category"]  # target variable

print("<wolse_dataset_normalized_one_hot - Monthly-Rent dataset DecisionTree Model Evaluation>")

for test_size in test_sizes:
    # Split dataset into train_set and test_set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=True,
        random_state=42,
    )

    # Decision Tree Modeling and Training
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)

    # Performing prediction on the test_set
    y_pred = dt_model.predict(X_test)

    # RMSE, R2 Score Calculation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Accuracy, Precision, Recall, F1 Score Calculation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1_ = f1_score(y_test, y_pred, average="weighted")

    # 10-Fold Cross Validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    repeated_kfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    stratified_kfold = StratifiedKFold(
        n_splits=10, shuffle=True, random_state=42)
    repeated_stratified_kfold = RepeatedStratifiedKFold(
        n_splits=10, n_repeats=10, random_state=42)

    # CVS Calculation
    cvs = cross_val_score(dt_model, X, y, cv=kfold)
    repeated_cvs = cross_val_score(
        dt_model, X, y, cv=repeated_kfold, scoring='accuracy')
    stratified_cvs = cross_val_score(
        dt_model, X, y, cv=stratified_kfold, scoring='accuracy')
    repeated_stratified_cvs = cross_val_score(
        dt_model, X, y, cv=repeated_stratified_kfold, scoring='accuracy')

    # Bootstrap Method
    n_iterations = 1000
    n_size = int(len(X) * 0.5)
    bootstrap_scores = []

    for _ in range(n_iterations):
        X_bs, y_bs = resample(X, y, n_samples=n_size, random_state=42)
        dt_model.fit(X_bs, y_bs)
        score = dt_model.score(X_test, y_test)
        bootstrap_scores.append(score)

    bootstrap_mean = np.mean(bootstrap_scores)

    # Display Result
    print(f"\nTest Size: {test_size}")
    print(f"1. Mean_Absolute_Error: {mae}, R2_Score: {r2}\n")

    # CVS values, CVS Average value comparison, Bootstrap, and Bootstrap Mean
    print("CVS Value")
    print("(1)KFold CVS\n", cvs)
    print()
    print("(2)Repeated KFold CVS\n", repeated_cvs)
    print()
    print("(3)Stratified KFold CVS\n", stratified_cvs)
    print()
    print("(4)Repeated Stratified KFold CVS\n", repeated_stratified_cvs)
    print()
    # Print the first 10 scores for brevity
    print("(5) Bootstrap Method\n", bootstrap_scores[:10], "...")
    print()
    print("2. CVS Mean Comparison")
    print(f"KFold CVS Mean: {cvs.mean()}")
    print(f"Repeated KFold CVS Mean: {repeated_cvs.mean()}")
    print(f"Stratified KFold CVS Mean: {stratified_cvs.mean()}")
    print(
        f"Repeated Stratified KFold CVS Mean: {repeated_stratified_cvs.mean()}")
    print(f"Bootstrap Mean: {bootstrap_mean}")

    # Show Accuracy, Precision, Recall, F1 Score
    print(f"Decision_tree_Accuracy: {accuracy}")
    print(f"Decision_tree_Precision: {precision}")
    print(f"Decision_tree_Recall: {recall}")
    print(f"Decision_tree_F1 Score: {f1_}")
    print(classification_report(y_test, y_pred))

    # confusion matrix (including all labels)
    labels = np.arange(len(criterion_labels))
    cm = confusion_matrix(y_test, y_pred, labels=labels).T
    print(f"cm(One-Hot method)\n{cm}\n")

    # Confusion Matrix Visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["very cheap", "cheap",
                     "appropriate", "expensive", "very expensive"],
        yticklabels=["very cheap", "cheap",
                     "appropriate", "expensive", "very expensive"],
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Wolse Deposit Confusion Matrix (one-hot method)")
    plt.show()

    # # 훈련된 결정 트리 모델의 트리 구조 시각화
    # plt.figure(figsize=(20, 10))  # 그래프의 크기 조절
    # plot_tree(
    #     dt_model,
    #     filled=True,
    #     rounded=True,
    #     class_names=["very cheap", "cheap", "appropriate", "expensive", "very expensive"],
    #     feature_names=X_train.columns,
    # )
    # plt.title("Decision Tree Visualization")
    # plt.show()

    # 하이퍼파라미터 그리드 설정
    param_grid = {
        "max_depth": [3, 4, 5, 6, 7, 8, 10, 12, 15],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "criterion": ["gini", "entropy"],
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(
        estimator=dt_model,
        param_grid=param_grid,
        cv=5,
        verbose=1,
        scoring="accuracy",
    )

    # 그리드 탐색 실행
    grid_search.fit(X_train, y_train)

    # 최적의 하이퍼파라미터와 그 성능 출력
    print("Best parameters:", grid_search.best_params_)
    print(
        "Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    # 최적의 모델로 예측 및 평가
    best_dt = grid_search.best_estimator_
    y_pred = best_dt.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 최적화된 모델의 confusion matrix 시각화
    cm_optimized = confusion_matrix(y_test, y_pred, labels=labels).T
    print(f"cm_optimized(one-hot method)\n{cm_optimized}\n")
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cm_optimized,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["very cheap", "cheap",
                     "appropriate", "expensive", "very expensive"],
        yticklabels=["very cheap", "cheap",
                     "appropriate", "expensive", "very expensive"],
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Optimized Decision Tree Confusion Matrix")
    plt.show()
