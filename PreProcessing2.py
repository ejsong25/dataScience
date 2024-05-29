import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

# plot 한글 깨짐 방지 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

"""
전세 데이터셋을 사용.
target(deposit(보증금)) 에 각 feaeture 가 미치는 영향 확인
"""
df = pd.read_csv("jeonse_dataset.csv")
pd.set_option("display.max_seq_items", None)

# 한글 칼럼명을 영어 칼럼명으로 변환
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
# 데이터 정규화 (stand scaling 사용)
df["contract_area_m2"] = scaler.fit_transform(df[["contract_area_m2"]])
df["deposit"] = scaler.fit_transform(df[["deposit"]])
df["building_age"] = scaler.fit_transform(df[["building_age"]])

X = df.iloc[:, 0:6]  # 'Road_condition' ~ 'Construction_year' columns
y = df.iloc[:, -1]  # 'Deposit' column

# 동 제외
"""
Importance scoring, corr heatmap 등을 제작할 때,
모든 "동" 을 포함하기보단 "동"은 제외하고 다른 특성들만 우선 적용
"""
df_with_out_district = pd.concat([X, y], axis=1)

model = ExtraTreesRegressor()
model.fit(X, y)

print(f"[전세 데이터]: Feature Importance (동 컬럼 제외): {model.feature_importances_}")

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
print(f"[전세 데이터]: Feature Importance (동 컬럼 제외): {feat_importances}")

plt.figure(figsize=(10, 6))
plt.title("[전세 데이터]: Feature Importance 시각화 (동 컬럼 제외)")
feat_importances.nlargest(6).plot(kind="barh")
plt.show()

corrmat = df_with_out_district.corr()

top_corr_features = corrmat.index
plt.figure(figsize=(15, 15))
plt.title("[전세 데이터]: Correlation Matrix 시각화 (동 컬럼 제외)")
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

# Only 동
"""
모든 "동" 특성들을 이용하여 상관관계 파악. 어느 동이 평균적으로 비싼지 등 판단 가능
"""
df_only_distrcit = df.drop(X.columns, axis=1)
corrmat = df_only_distrcit.corr()

# ----------------------------------------------------------------------
# "동" 컬럼만 추출하여 지역 점수 생성
'''
'동'과 보증금 간의 상관관계 계산, 
보증금 - 보증금 상관관계는 1이므로 drop
상관관계를 표준화 하여 지역 점수로 반영
'''

# 각 '동'이 보증금에 미치는 상관 관계 계산, 표준화 => 지역점수

district_corr = df_only_distrcit.corr()["deposit"].drop("deposit")
district_index = district_corr.index  # 지역명 인덱스 저장

district_corr = district_corr.values.reshape(-1, 1)
district_scores = StandardScaler().fit_transform(district_corr)

# column : 동, row: district score
district_scores_df = pd.DataFrame(district_scores, index=district_index, columns=["district_score"]).T
print(district_scores_df)


# 지역 점수를 원래 데이터프레임에 추가
# 각 행에서 True인 값의 컬럼명을 찾아 해당하는 지역 점수를 새 컬럼으로 추가

def get_district_score(row):
    for col in district_index:
        if row[col] == 1:
            return district_scores_df[col].values[0]
    return 0  # 모든 값이 False인 경우

df["district_score"] = df.apply(get_district_score, axis=1)

# 원핫 인코딩된 컬럼 drop
df = df.drop(columns=district_index)
deposit = df.pop("deposit")
df['deposit'] = deposit
print(df.head())

# 지역 점수를 포함한 상관관계 히트맵 생성
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

# normalize된 전세 데이터를 csv로 저장
# df.to_csv('jeonse_dataset_normalized.csv', index=False, encoding='utf-8-sig')

"""
월세 데이터셋을 사용.
target(monthly_rent_bill(월세금)) 에 각 feaeture 가 미치는 영향 확인
"""
df = pd.read_csv("wolse_dataset.csv")

# 한글 칼럼명을 영어 칼럼명으로 변환
df = convert_to_english(df)

# 데이터 정규화 (stand scaling 사용)
df["contract_area_m2"] = scaler.fit_transform(df[["contract_area_m2"]])
df["building_age"] = scaler.fit_transform(df[["building_age"]])
df["deposit"] = scaler.fit_transform(df[["deposit"]])
df["monthly_rent_bill"] = scaler.fit_transform(df[["monthly_rent_bill"]])

X = df.iloc[:, 0:7]  # 'Road_condition' ~ 'Contract_month' columns
y = df.iloc[:, -1]  # 'Monthly_rent_bill' column

# 동 제외
"""
Importance scoring, corr heatmap 등을 제작할 때
모든 "동" 을 포함하기보단, "동"은 제외하고 다른 특성들만 우선 적용
"""
df_with_out_district = pd.concat([X, y], axis=1)

model = ExtraTreesRegressor()
model.fit(X, y)

print(f"[월세 데이터]: Feature Importance (동 컬럼 제외)\n{model.feature_importances_}")

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
plt.title("[월세 데이터]: Feature Importance 시각화 (동 컬럼 제외)")
feat_importances.nlargest(6).plot(kind="barh")
plt.show()

corrmat = df_with_out_district.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15, 15))
plt.title("[월세 데이터]: Correlation Matrix 시각화 (동 컬럼 제외)")
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

# only 동
"""
모든 "동" 특성들을 이용하여 상관관계 파악. 어느 동이 평균적으로 비싼지 등 판단 가능
"""
df_only_distrcit = df.drop(X.columns, axis=1)

corrmat = df_only_distrcit.corr()

# ----------------------------------------------------------------------
# "동" 컬럼만 추출하여 지역 점수 생성
'''
'동'과 월세금 간의 상관관계 계산, 
월세금 - 월세금 상관관계는 1이므로 drop
상관관계를 표준화 하여 지역 점수로 반영
'''

# 각 '동'이 월세금 미치는 상관 관계 계산, 표준화 => 지역점수

district_corr = df_only_distrcit.corr()["monthly_rent_bill"].drop("monthly_rent_bill")
district_index = district_corr.index  # 지역명 인덱스 저장

district_corr = district_corr.values.reshape(-1, 1)
district_scores = StandardScaler().fit_transform(district_corr)

# column : 동, row: district score
district_scores_df = pd.DataFrame(district_scores, index=district_index, columns=["district_score"]).T
print("district score dataframe")
print(district_scores_df)

# 지역 점수를 원래 데이터프레임에 추가
# 각 행에서 True인 값의 컬럼명을 찾아 해당하는 지역 점수를 새 컬럼으로 추가

def get_district_score(row):
    for col in district_index:
        if row[col] == 1:
            return district_scores_df[col].values[0]
    return 0  # 모든 값이 False인 경우

df["district_score"] = df.apply(get_district_score, axis=1)

# 원핫 인코딩된 컬럼 drop
df = df.drop(columns=district_index)
cost = df.pop("monthly_rent_bill")
df['monthly_rent_bill'] = cost

print("renewed original dataset")
print(df.head())

# 지역 점수를 포함한 상관관계 히트맵 생성
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

# normalize된 월세 데이터를 csv로 저장
# df.to_csv("wolse_dataset_normalized.csv", index=False, encoding="utf-8-sig")
