from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from datetime import datetime
import re
import pandas as pd

'''
Replace DirtyData into null
'''
df = pd.read_csv('data/seongnam_dataset.csv', na_values='-')
pd.set_option('display.max_seq_items', None)

'''
특성 이름 영어로 변경
'''
column_mapping = {
    '시군구': 'district',
    '번지': 'lot_number',
    '도로조건': 'road_condition',
    '계약면적(㎡)': 'contract_area_m2',
    '전월세구분': 'lease_type',
    '계약년월': 'contract_year_month',
    '계약일': 'contract_day',
    '보증금(만원)': 'deposit',
    '월세금(만원)': 'monthly_rent_bill',
    '건축년도': 'construction_year',
    '도로명': 'road_name',
    '계약기간': 'contract_period',
    '계약구분': 'contract_type',
    '갱신요구권 사용': 'renewal_right_used',
    '종전계약 보증금(만원)': 'previous_contract_deposit',
    '종전계약 월세(만원)': 'previous_contract_monthly_rent_bill',
    '주택유형': 'housing_type'
}

# 컬럼 이름 변경
df = df.rename(columns=column_mapping)

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

ver2. 05/24

전처리 2차 회의 후, 시군구는 일단 사용하지 않기로 결정.
추후에 동별 추이 및 평계를 위해 추가적 디벨롭 할 때 다시 사용하기로.
'''
indices_to_use = [2, 3, 4, 5, 7, 8, 9, 10, 11]
df = df.iloc[:, indices_to_use]

print('[Sample data after dropping useless columns]\n')
print(df.head(10))
print('\n====================================================================================================================\n')

'''
# 시군구 동 추출---------------------------------------------

ver2. 05/24

전처리 2차 회의 후, 시군구는 일단 사용하지 않기로 결정.
추후에 동별 추이 및 평계를 위해 추가적 디벨롭 할 때 다시 사용하기로.
'''

# df['시군구'] = df['시군구'].apply(lambda x: x.split()[-1])

# 라벨인코딩-통계 자료로 활용한다면 필요한 작업인가?
# le = LabelEncoder()
# df['시군구'] = le.fit_transform(df['시군구'])


'''
보증금(만원): 콤마 제거
'''
df['deposit'] = df['deposit'].fillna('0').apply(
    lambda x: int(x.replace(',', '')))

'''
전월세구분 wrong data:

보증금 == 0
월세 구분인데 월세금 == 0
전세 구분인데 월세금 != 0

데이터 제거
'''
index_to_drop = df[(df['deposit'] == 0)].index
df.drop(index_to_drop, inplace=True)

index_to_drop = df[(df['lease_type'] == '월세') &
                   (df['monthly_rent_bill'] == 0)].index
df.drop(index_to_drop, inplace=True)

index_to_drop = df[(df['lease_type'] == '전세') &
                   (df['monthly_rent_bill'] != 0)].index
df.drop(index_to_drop, inplace=True)

'''
전월세구분: ['전세', '월세'] -> 우위가 있는 게 아니기에,
one hot encoding 진행
'''
df_encoded = pd.get_dummies(df['lease_type'])
df = pd.concat([df, df_encoded], axis=1)
df = df.drop(columns=['lease_type'])

'''
건축연식 = 현재 년도 - 건축년도
숫자가 작을수록 좋음
'''
df = df.dropna(subset=['construction_year'])
df['building_age'] = (datetime.now().year -
                      df['construction_year']).astype(int)
df = df.drop(columns=['construction_year'])


'''
계약기간: 202401 ~ 202601
1. '~` 를 기준으로 년도 두 개를 추출한다. (2024, 2026)
2. 종료년도에서 시작년도를 뺀다
3. null value 들은 평균 값으로 채워준다.
'''
# 반년 단위 계약은 없었는지?
df['contract_period'] = df['contract_period'].apply(
    lambda x: int(x.split('~')[1][:4])if pd.notna(x) else None) - df['contract_period'].apply(
    lambda x: int(x.split('~')[0][:4]) if pd.notna(x) else None)

df['contract_period'] = df['contract_period'].fillna(
    round(df['contract_period'].mean()))
df['contract_period'] = df['contract_period'].astype(int)

df['contract_year'] = df['contract_year_month']//100
df['contract_month'] = df['contract_year_month'] % 100
df = df.drop(columns=['contract_year_month'])

'''
도로명: ㅁㅁㅁㅁ길ㅇㅇㅇ번지 (ㅁ: 문자, ㅇ: 정수)
1. 정수를 기준으로 두 파트로 나눈다. [ㅁㅁㅁㅁ길, ㅇㅇㅇ번지]
2. ㅇㅇㅇ번지까지 도로명을 구분하면 너무 많기에, ㅁㅁㅁ길로 통일해준다.
'''
df = df.dropna(subset=['road_name'])
pattern = r'(\D+)(\d+)?'

df['road_name'] = df['road_name'].apply(lambda x: re.match(
    pattern, x).group(1) if pd.notna(x) else None)


'''
24개 도로명 > 라벨 인코딩 진행
'''
le = LabelEncoder()
df['road_name'] = le.fit_transform(df['road_name'])


'''
도로조건: ['8m미만', '12m미만', '25m미만', '25m이상', ]: 매물과 인근한 도로의 넓이

넓이가 넓을수록 교통이 좋다고 판단.
따라서 one-hot encoding 이 아닌, ordinal-encoding 진행
'''
encoder = OrdinalEncoder(categories=[['8m미만', '12m미만', '25m미만', '25m이상', ]])
df['road_condition'] = encoder.fit_transform(df[['road_condition']])

'''
target Value 를 가장 끝으로 옮기기
'''
df_target = df[['deposit', 'monthly_rent_bill']]
df = df.drop(columns=['deposit', 'monthly_rent_bill'])
df = pd.concat([df, df_target], axis=1)


'''
월세, 전세 데이터 분리 후 저장
'''
df_js = df[df['전세'] == True]
df_js = df_js.drop(columns=['전세', '월세', 'monthly_rent_bill'])


df_ws = df[df['월세'] == True]
df_ws = df_ws.drop(columns=['월세', '전세'])


# df_ws.to_csv('wolse_dataset.csv', index=False, encoding='utf-8-sig')
# df_js.to_csv('jeonse_dataset.csv', index=False, encoding='utf-8-sig')
