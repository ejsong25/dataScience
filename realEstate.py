from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from datetime import datetime
import re
import pandas as pd

'''
Replace DirtyData into null
'''
df = pd.read_csv('data/sujung.csv', na_values='-')
pd.set_option('display.max_seq_items', None)

'''
기본 통계 데이터
'''
print('\n====================================================================================================================\n')
print('[df.describe()]\n')
print(df.describe())
print('\n====================================================================================================================\n')

print('[df.info()]\n')
print(df.info())
print('\n====================================================================================================================\n')

'''
column별 null data 수
'''
print('[Column 별 null data 수]\n')
print(df.isnull().sum())
print('\n====================================================================================================================\n')

'''
[유의미 데이터 추출]
Drop column - 번지, 계약일, 계약구분 부터 모든 칼럼
사용 column - 시군구, 도로조건, 계약면적, 전월세구분, 계약년월, 보증금(만원), '월세금(만원)', 건축년도, 도로명, 계약기간
'''
indices_to_use = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11]
df = df.iloc[:, indices_to_use]

print('[Sample data after dropping useless columns]\n')
print(df.head(10))
print('\n====================================================================================================================\n')

# 시군구 동 추출---------------------------------------------
df['시군구'] = df['시군구'].apply(lambda x: x.split()[-1])

# 라벨인코딩-통계 자료로 활용한다면 필요한 작업인가?
le = LabelEncoder()
df['시군구'] = le.fit_transform(df['시군구'])
# print(df.head())
# ----------------------------------------------------------

'''
보증금(만원): 콤마 제거
'''
df['보증금(만원)'] = df['보증금(만원)'].apply(lambda x: int(x.replace(',', '')))

'''
전월세구분 wrong data:

월세 구분인데 월세금 == 0
전세 구분인데 월세금 != 0

데이터 제거
'''
index_to_drop = df[(df['전월세구분'] == '월세') & (df['월세금(만원)'] == 0)].index
df.drop(index_to_drop, inplace=True)

index_to_drop = df[(df['전월세구분'] == '전세') & (df['월세금(만원)'] != 0)].index
df.drop(index_to_drop, inplace=True)

'''
전월세구분: ['전세', '월세'] -> 우위가 있는 게 아니기에,
one hot encoding 진행
'''
df_encoded = pd.get_dummies(df['전월세구분'])
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(columns=['전월세구분'])

'''
건축연식 = 현재 년도 - 건축년도
숫자가 작을수록 좋음
'''
df = df.dropna(subset=['건축년도'])
df['건물연식'] = (datetime.now().year - df['건축년도']).astype(int)
df = df.drop(columns=['건축년도'])
# print(df.head(10))

'''
계약기간: 202401 ~ 202601
1. '~` 를 기준으로 년도 두 개를 추출한다. (2024, 2026)
2. 종료년도에서 시작년도를 뺀다
3. null value 들은 평균 값으로 채워준다. 
'''

df['계약기간'] = df['계약기간'].apply(
    lambda x: int(x.split('~')[1][:4])if pd.notna(x) else None) - df['계약기간'].apply(
    lambda x: int(x.split('~')[0][:4]) if pd.notna(x) else None)

df['계약기간'] = df['계약기간'].fillna(round(df['계약기간'].mean()))
df['계약기간'] = df['계약기간'].astype(int)

df['계약년도'] = df['계약년월']//100
df['계약월'] = df['계약년월'] % 100
df = df.drop(columns=['계약년월'])
# print(df.head(10))

'''
도로명: ㅁㅁㅁㅁ길ㅇㅇㅇ번지 (ㅁ: 문자, ㅇ: 정수)
1. 정수를 기준으로 두 파트로 나눈다. [ㅁㅁㅁㅁ길, ㅇㅇㅇ번지]
2. ㅇㅇㅇ번지까지 도로명을 구분하면 너무 많기에, ㅁㅁㅁ길로 통일해준다.
'''
df = df.dropna(subset=['도로명'])
pattern = r'(\D+)(\d+)?'

df['도로명'] = df['도로명'].apply(lambda x: re.match(
    pattern, x).group(1) if pd.notna(x) else None)
print(df.head(10))

print("\n도로명 null drop 이후 도로조건 null 확인========================")
print(df.isnull().sum())

'''
24개 도로명 > 라벨 인코딩 진행
'''
le = LabelEncoder()
df['도로명'] = le.fit_transform(df['도로명'])

'''
도로조건: ['8m미만', '12m미만', '25m미만', '25m이상', ]: 매물과 인근한 도로의 넓이

넓이가 넓을수록 교통이 좋다고 판단.
따라서 one-hot encoding 이 아닌, ordinal-encoding 진행
'''
encoder = OrdinalEncoder(categories=[['8m미만', '12m미만', '25m미만', '25m이상', ]])
df['도로조건'] = encoder.fit_transform(df[['도로조건']])

'''
1차 전처리 후 출력
'''
print('[Sample data]\n')
print(df.head())
print('\n====================================================================================================================\n')

# column별 null data 수
print('[Column 별 null data 수]\n')
print(df.isnull().sum())
print('\n====================================================================================================================\n')

# numerical data 확인용
print(df.info())

# Scaling - 모든 numerical data에 대해 진행?
# 우선 계약면적, 보증금, 월세금, 계약기간, 건물 연식만 진행
scaler = StandardScaler()
df['계약면적(㎡)'] = scaler.fit_transform(df[['계약면적(㎡)']])
df['보증금(만원)'] = scaler.fit_transform(df[['보증금(만원)']])
df['월세금(만원)'] = scaler.fit_transform(df[['월세금(만원)']])
df['계약기간'] = scaler.fit_transform(df[['계약기간']])
df['건물연식'] = scaler.fit_transform(df[['건물연식']])

print(df.head(10))


