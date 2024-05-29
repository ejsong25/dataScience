import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option("display.max_seq_items", None)

# 전세,월세 데이터 로드
jeonse_data = pd.read_csv("jeonse_dataset_normalized.csv")
wolse_data = pd.read_csv("wolse_dataset_normalized.csv")

# Display the first 5 rows of the each dataset
print(jeonse_data.head())
print(wolse_data.head())

# 각 데이터를 8:2로 나누어 train, test 데이터로 사용
jeonse_train, jeonse_test = train_test_split(
    jeonse_data,
    test_size=0.2,
    stratify=jeonse_data["deposit"],
)
wolse_train, wolse_test = train_test_split(
    wolse_data,
    test_size=0.2,
    stratify=wolse_data["monthly_rent"],
)
