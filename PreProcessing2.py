import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd

'''
전세 데이터셋을 사용.
target(deposit(보증금)) 에 각 feaeture 가 미치는 영향 확인
'''
df = pd.read_csv('jeonse_dataset.csv')

# 한글 칼럼명을 영어 칼럼명으로 변환
def convert_to_english(dataframe):
    english_columns = ["Galhyeon", "Godeung", "Geumgwang", "Geumto", "Dandae",
                       "Dochon", "Bokjeong", "Sasong", "Sanseong", "Sangdaewon",
                       "Sangjeok", "Seongnam", "Sujin", "Siheung", "Sinchon",
                       "Sinhung", "Simgok", "Yangji", "Yeosu", "Oya",
                       "Eunhaeng", "Jungang", "Changok", "Taepyeong", "Hadaewon"]
    
    # 한글 칼럼명을 영어 칼럼명으로 변환
    dataframe.columns = [english_columns.pop(0) if col.endswith('동') else col for col in dataframe.columns]
    return df

df=convert_to_english(df)

scaler = StandardScaler()

df['contract_area_m2'] = scaler.fit_transform(df[['contract_area_m2']])
df['deposit'] = scaler.fit_transform(df[['deposit']])
df['building_age'] = scaler.fit_transform(df[['building_age']])

X = df.iloc[:, 0:6]
y = df.iloc[:, -1]

'''
Importance scoring, corr heatmap 등을 제작할 때, 모든 "동" 을 포함하기보단, 
"동"은 제외하고 다른 특성들만 우선 적용
'''
df_with_out_district = pd.concat([X, y], axis=1)

model = ExtraTreesRegressor()
model.fit(X, y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()


corrmat = df_with_out_district.corr()

top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

'''
모든 "동" 특성들을 이용하여 상관관계 파악.
어느 동이 평균적으로 비싼지 등 판단 가능
'''
df_only_distrcit = df.drop(X.columns, axis=1)

corrmat = df_only_distrcit.corr()

top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

# df.to_csv('jeonse_dataset_normalized.csv', index=False, encoding='utf-8-sig')

'''
월세 데이터셋을 사용.
target(monthly_rent_bill(월세금)) 에 각 feaeture 가 미치는 영향 확인
'''
df = pd.read_csv('wolse_dataset.csv')

# 한글 칼럼명을 영어 칼럼명으로 변환
df=convert_to_english(df)

df['contract_area_m2'] = scaler.fit_transform(df[['contract_area_m2']])
df['building_age'] = scaler.fit_transform(df[['building_age']])
df['deposit'] = scaler.fit_transform(df[['deposit']])
df['monthly_rent_bill'] = scaler.fit_transform(df[['monthly_rent_bill']])

X = df.iloc[:, 0:7]
y = df.iloc[:, -1]


'''
Importance scoring, corr heatmap 등을 제작할 때, 모든 "동" 을 포함하기보단, 
"동"은 제외하고 다른 특성들만 우선 적용
'''
df_with_out_district = pd.concat([X, y], axis=1)

model = ExtraTreesRegressor()
model.fit(X, y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(6).plot(kind='barh')
plt.show()


corrmat = df_with_out_district.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

'''
모든 "동" 특성들을 이용하여 상관관계 파악.
어느 동이 평균적으로 비싼지 등 판단 가능
'''
df_only_distrcit = df.drop(X.columns, axis=1)

corrmat = df_only_distrcit.corr()

top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

df.to_csv('wolse_dataset_normalized.csv', index=False, encoding='utf-8-sig')
