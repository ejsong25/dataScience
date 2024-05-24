import re
import pandas as pd

# 도로명 주소 test
'''
# 주어진 문자열들
texts = ["시민로6번길", "시민로6번길", "시민로67번길", "신촌로"]

# 정규 표현식을 사용하여 문자열, 정수, 문자열로 나누기
pattern = r'(\D+)(\d+)?(\D+)?'

for text in texts:
    result = re.match(pattern, text)
    if result:
        string_part = result.group(1)

        print("문자열:", string_part)

        print()
    else:
        print("정규 표현식에 맞는 패턴을 찾을 수 없습니다.")
'''

# 도로조건 언제 null test

# 도로명, 도로조건 관계성 test
'''
df = pd.read_csv('data/jungwon.csv', na_values='-')

indices_to_use = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11]
df = df.iloc[:, indices_to_use]
pd.set_option('display.max_seq_items', None)

print(df['도로조건'].isnull().sum())

1. [[도로명이 null일때]]

도로명 null값을 drop하면 일부 처리되긴하지만 도로조건 null이 남아있음

- 처리
도로명 null 도로조건 ! null : 10개
도로명 null 도로조건 null : 1653개

- 미처리
도로명 ! null 도로조건 null : 1104개

-> 도로명 null data drop하는 과정에서 도로조건 null data가 처리 된것이 아님

result1 = df[df['도로명'].isnull() & df['도로조건'].notnull()]
result2 = df[df['도로명'].isnull() & df['도로조건'].isnull()]
result3 = df[df['도로명'].notnull() & df['도로조건'].isnull()]

# 결과 출력
print(result1.isnull().sum())
print(result2.isnull().sum())
print(result3.isnull().sum())
'''

# 건축년도와 도로조건 관계성 test
'''
도로명으로 미처리된 도로조건 null데이터는 언제 삭제되었는가
2. [[건축년도가 null일때]]

result3에 따르면 
도로명 !null 도로조건 null 인경우
건축년도 null 개수 동일

도로명 !null, 건축년도 null, 도로조건 null : 1104개

-> 도로명 null drop해서 해결되지 않은 도로조건 null은 건축년도 null data drop시 처리됨

result4 = df[df['건축년도'].isnull() & df['도로조건'].isnull() & df['도로명'].notnull()]
print(result4.isnull().sum())
'''
