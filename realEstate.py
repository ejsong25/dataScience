import pandas as pd

# Replace DirtyData into null
df = pd.read_csv('data/jungwon.csv', na_values='-')
pd.set_option('display.max_seq_items', None)

# 기본 통계 데이터
# print(df.describe())
# print(df.info())

# column별 null data 수
print(df.isnull().sum())
print('------------------------------------------')

# [유의미 데이터 추출]
# Drop column - 시군구, 번지, 계약일, 계약구분 부터 모든 칼럼
# 사용 column - 도로조건, 계약면적, 전월세구분, 계약년월, 보증금(만원), '월세금(만원)', 건축년도, 도로명, 계약기간
indices_to_use = [2, 3, 4, 5, 7, 8, 9, 10, 11]
df = df.iloc[:, indices_to_use]
print(df.head())
print('------------------------------------------')

# column별 null data 수
print(df.isnull().sum())
print('------------------------------------------')

# Drop rows - 