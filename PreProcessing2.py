import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

# Preventing Koerean crush in plots
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

"""
Use jeonse dataset
target(deposit(보증금)) 에 각 feaeture 가 미치는 영향 확인
"""
df = pd.read_csv("jeonse_dataset.csv")
pd.set_option("display.max_seq_items", None)


# Translate Feature names into English
def convert_to_english(dataframe):
    english_columns = [
        "Galhyeon",
        "Godeung",
        "Geumgwang",
        "Geumto",
        "Dandae",
        "Dochon",
        "Bokjeong",
        "Sasong",
        "Sanseong",
        "Sangdaewon",
        "Sangjeok",
        "Seongnam",
        "Sujin",
        "Siheung",
        "Sinchon",
        "Sinhung",
        "Simgok",
        "Yangji",
        "Yeosu",
        "Oya",
        "Eunhaeng",
        "Jungang",
        "Changok",
        "Taepyeong",
        "Hadaewon",
    ]
    dataframe.columns = [
        english_columns.pop(0) if col.endswith("동") else col
        for col in dataframe.columns
    ]
    return df


df = convert_to_english(df)

scaler = StandardScaler()
# Data Normalization (Using Stand Scaling)
df["contract_area_m2"] = scaler.fit_transform(df[["contract_area_m2"]])
df["building_age"] = scaler.fit_transform(df[["building_age"]])

X = df.iloc[:, 0:4]  # 'Road_condition' ~ 'building_age' columns
y = df.iloc[:, -1]  # 'Deposit' column

# Excluding district
"""
When creating importance scoring, correlation heatmaps, etc.,
prioritize applying other features excluding 'district' rather than including all 'district'.
"""
df_with_out_district = pd.concat([X, y], axis=1)

model = ExtraTreesRegressor()
model.fit(X, y)

print(f"[jeonse data]: Feature Importance (district column excluded): {model.feature_importances_}")

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(f"[jeonse data]: Feature Importance (district column excluded): {feat_importances}")

plt.figure(figsize=(10, 6))
plt.title("[jeonse data]: Feature Importance Visualization (district column excluded)")
feat_importances.nlargest(6).plot(kind="barh")
plt.show()

corrmat = df_with_out_district.corr()

top_corr_features = corrmat.index
plt.figure(figsize=(15, 15))
plt.title("[jeonse data]: Correlation Matrix Visualization (district column excluded)")
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

# Only district
"""
Using all 'district' features to determine correlations.
It allows assessing which district is generally more expensive on average.
"""
df_only_distrcit = df.drop(X.columns, axis=1)
corrmat = df_only_distrcit.corr()

# ----------------------------------------------------------------------
# Extracting only the 'district' column to create a "district_scores"
"""
Calculating the correlation between 'district' and deposit.
Since (deposit-deposit correlation = 1), drop.
Standardize the correlation to reflect to district_scores.
"""
''' Calculating the correlation between each 'district' and deposit,
After standardizing => district_scores'''
district_corr = df_only_distrcit.corr()["deposit"].drop("deposit")
district_index = district_corr.index  # Saving district name indices

district_corr = district_corr.values.reshape(-1, 1)
district_scores = StandardScaler().fit_transform(district_corr)

# column : district, row: district score
district_scores_df = pd.DataFrame(
    district_scores, index=district_index, columns=["district_score"]
).T
print(district_scores_df)


# Add "district_scores" to the original DataFrame
# Finding the column names where values are True in each row and adding the corresponding district_scores as a new column

def get_district_score(row):
    for col in district_index:
        if row[col] == 1:
            return district_scores_df[col].values[0]
    return 0  # when all values are False


df["district_score"] = df.apply(get_district_score, axis=1)

# Drop columns that have been one-hot encoded
df = df.drop(columns=district_index)
deposit = df.pop("deposit")
df["deposit"] = deposit
print(df.head())

# Creating a correlation heatmap including district_scores
corrmat_district_score = df[["district_score", "deposit"]].corr()
plt.figure(figsize=(6, 6))
plt.title("[전세 데이터]: Correlation Matrix 시각화 (지역 점수 포함)")
sns.heatmap(corrmat_district_score, annot=True, cmap="RdYlGn")
plt.show()
# ---------------------------------------------------------------------


top_corr_features = corrmat.index
plt.figure(figsize=(15, 15))
plt.title("[전세 데이터]: Correlation Matrix 시각화 (동 비교)")
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

# Saving normalized jeonse data into csv file
df.to_csv(
    "jeonse_dataset_normalized.csv",
    mode="w",
    index=False,
    encoding="utf-8-sig",
)

"""
월세 데이터셋을 사용.
target(monthly_rent_bill(월세금)) 에 각 feaeture 가 미치는 영향 확인
"""
df = pd.read_csv("wolse_dataset.csv")

# Translate Feature names into English
df = convert_to_english(df)

# Data Normalization (Using Stand Scaling)
df["contract_area_m2"] = scaler.fit_transform(df[["contract_area_m2"]])
df["building_age"] = scaler.fit_transform(df[["building_age"]])
df["deposit"] = scaler.fit_transform(df[["deposit"]])

# Comment on normalizing the target variable
# df["monthly_rent_bill"] = scaler.fit_transform(df[["monthly_rent_bill"]])

X = df.iloc[:, 0:5]  # 'Road_condition' ~ 'Contract_month' columns
y = df.iloc[:, -1]  # 'Monthly_rent_bill' column

# Extract district
"""
When creating Importance scoring and corr heatmap,
excluding 'district' and applying other features first rather than including all 'districts'
"""
df_with_out_district = pd.concat([X, y], axis=1)

model = ExtraTreesRegressor()
model.fit(X, y)

print(f"[wolse data]: Feature Importance (district column excluded)\n{model.feature_importances_}")

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
plt.title("[Wolse Data]: Feature Importance Visualization (district column excluded)")
feat_importances.nlargest(6).plot(kind="barh")
plt.show()

corrmat = df_with_out_district.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15, 15))
plt.title("[Wolse Data]]: Correlation Matrix Visualization (district column excluded)")
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

# only district
"""
Using all 'district' features to determine correlations.
It allows assessing which district is generally more expensive on average.
"""
df_only_distrcit = df.drop(X.columns, axis=1)

corrmat = df_only_distrcit.corr()

# ----------------------------------------------------------------------
# Extracting only the 'district' column to create a "district_scores"
"""
Calculating the correlation between 'district' and monthly rent bill.
Since (monthly-rent bill-monthly=rent bill correlation = 1), drop.
Standardize the correlation to reflect to district_scores.
"""
''' Calculating the correlation between each 'district' and monthly rent bill,
After standardizing => district_scores'''

district_corr = df_only_distrcit.corr()["monthly_rent_bill"].drop("monthly_rent_bill")
district_index = district_corr.index  # Saving district name indices

district_corr = district_corr.values.reshape(-1, 1)
district_scores = StandardScaler().fit_transform(district_corr)

# column : district, row: district score
district_scores_df = pd.DataFrame(
    district_scores, index=district_index, columns=["district_score"]
).T
print("district score dataframe")
print(district_scores_df)

# Add "district_scores" to the original DataFrame
# Finding the column names where values are True in each row and adding the corresponding district_scores as a new column


def get_district_score(row):
    for col in district_index:
        if row[col] == 1:
            return district_scores_df[col].values[0]
    return 0  # when all values are False


df["district_score"] = df.apply(get_district_score, axis=1)

# Drop columns that have been one-hot encoded
df = df.drop(columns=district_index)
cost = df.pop("monthly_rent_bill")
df["monthly_rent_bill"] = cost

print("renewed original dataset")
print(df.head())

# Creating a correlation heatmap including district_scores
corrmat_district_score = df[["district_score", "monthly_rent_bill"]].corr()
plt.figure(figsize=(6, 6))
plt.title("[월세 데이터]: Correlation Matrix 시각화 (지역 점수 포함)")
sns.heatmap(corrmat_district_score, annot=True, cmap="RdYlGn")
plt.show()
# ---------------------------------------------------------------------

top_corr_features = corrmat.index
plt.figure(figsize=(15, 15))
plt.title("[월세 데이터]: Correlation Matrix 시각화 (동 비교)")
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

# Saving normalized wolse data into csv file
df.to_csv(
    "wolse_dataset_normalized.csv",
    mode="w",
    index=False,
    encoding="utf-8-sig",
)
