import re

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
