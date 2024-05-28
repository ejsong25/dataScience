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

scaler = StandardScaler()

df['contract_area_m2'] = scaler.fit_transform(df[['contract_area_m2']])
df['deposit'] = scaler.fit_transform(df[['deposit']])
df['building_age'] = scaler.fit_transform(df[['building_age']])

X = df.iloc[:, 0:7]
y = df.iloc[:, -1]


model = ExtraTreesRegressor()
model.fit(X, y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()

df.to_csv('jeonse_dataset_normalized.csv', index=False, encoding='utf-8-sig')

'''
월세 데이터셋을 사용.
target(monthly_rent_bill(월세금)) 에 각 feaeture 가 미치는 영향 확인
'''
df = pd.read_csv('wolse_dataset.csv')

df['contract_area_m2'] = scaler.fit_transform(df[['contract_area_m2']])
df['building_age'] = scaler.fit_transform(df[['building_age']])
df['deposit'] = scaler.fit_transform(df[['deposit']])
df['monthly_rent_bill'] = scaler.fit_transform(df[['monthly_rent_bill']])

X = df.iloc[:, 0:8]
y = df.iloc[:, -1]


model = ExtraTreesRegressor()
model.fit(X, y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
plt.show()


df.to_csv('wolse_dataset_normalized.csv', index=False, encoding='utf-8-sig')
