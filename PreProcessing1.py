from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from datetime import datetime
import re
import pandas as pd

"""
Replace DirtyData into null
"""
df = pd.read_csv("data/seongnam_dataset.csv", na_values="-")
pd.set_option("display.max_seq_items", None)

"""
Translate Feature names into English
"""
column_mapping = {
    "시군구": "district",
    "번지": "lot_number",
    "도로조건": "road_condition",
    "계약면적(㎡)": "contract_area_m2",
    "전월세구분": "lease_type",
    "계약년월": "contract_year_month",
    "계약일": "contract_day",
    "보증금(만원)": "deposit",
    "월세금(만원)": "monthly_rent_bill",
    "건축년도": "construction_year",
    "도로명": "road_name",
    "계약기간": "contract_period",
    "계약구분": "contract_type",
    "갱신요구권 사용": "renewal_right_used",
    "종전계약 보증금(만원)": "previous_contract_deposit",
    "종전계약 월세(만원)": "previous_contract_monthly_rent_bill",
    "주택유형": "housing_type",
}

# Changing Column Name
df = df.rename(columns=column_mapping)

"""
Basic statistical data
"""
print(
    "\n====================================================================================================================\n"
)
print("[df.describe()]\n")
print(df.describe())
print(
    "\n====================================================================================================================\n"
)

print("[df.info()]\n")
print(df.info())
print(
    "\n====================================================================================================================\n"
)

"""
Number of null data per column
"""
print("[Column 별 null data 수]\n")
print(df.isnull().sum())
print(
    "\n====================================================================================================================\n"
)

"""
[Extracting meaningful data]
Drop column - 번지, 계약년월, 계약일, 도로명, and all columns after "계약구분" 
Using column -  시군구, 도로조건, 계약면적, 전월세구분, 보증금(만원), '월세금(만원)', 건축년도, 도로명, 계약기간
"""

"""
ver2. 05/24

After the second preprocessing meeting, 
we decided not to use the administrative district for now. 
It will be revisited for further development when analyzing trends and averages by neighborhood in the future.

ver3.  05/28 Adding "시군구" column 
"""

''' ver4. 05/30 "계약년월" drop - unnecessary for price prediction, 
It will be used for extracting statistical data later'''

indices_to_use = [0, 2, 3, 4, 7, 8, 9, 11]
df = df.iloc[:, indices_to_use]

print("[Sample data after dropping useless columns]\n")
print(df.head(10))
print(
    "\n====================================================================================================================\n"
)


"""
보증금(만원): Replacing comman
"""
df["deposit"] = df["deposit"].fillna("0").apply(lambda x: int(x.replace(",", "")))

"""
Type of Lease (Jeonse or Monthly Rent) wrong data:

deposit == 0
when type: wolse but (monthly rent bill) == 0
when type: jeonse but deposit != 0

Replacing data
"""
index_to_drop = df[(df["deposit"] == 0)].index
df.drop(index_to_drop, inplace=True)

index_to_drop = df[(df["lease_type"] == "월세") & (df["monthly_rent_bill"] == 0)].index
df.drop(index_to_drop, inplace=True)

index_to_drop = df[(df["lease_type"] == "전세") & (df["monthly_rent_bill"] != 0)].index
df.drop(index_to_drop, inplace=True)


"""
building age = (current year) - (year of construction)
The lower, the better
"""
df = df.dropna(subset=["construction_year"])
df["building_age"] = (datetime.now().year - df["construction_year"]).astype(int)
df = df.drop(columns=["construction_year"])


"""
Contract Period : 202401 ~ 202601
1. Extract two years based on '~`". (2024, 2026)
2. Subtract the start year from the end year
3. Fill Null values are with mean value.
"""
df["contract_period"] = df["contract_period"].apply(
    lambda x: int(x.split("~")[1][:4]) if pd.notna(x) else None
) - df["contract_period"].apply(
    lambda x: int(x.split("~")[0][:4]) if pd.notna(x) else None
)

df["contract_period"] = df["contract_period"].fillna(
    round(df["contract_period"].mean())
)
df["contract_period"] = df["contract_period"].astype(int)

# ver4. contract_year_month drop
'''
df["contract_year"] = df["contract_year_month"] // 100
df["contract_month"] = df["contract_year_month"] % 100
df = df.drop(columns=["contract_year_month"])
'''
"""
road_condition : ['8m미만', '12m미만', '25m미만', '25m이상', ]: Width of Road near the Property

Wider width is considered indicative of better transportation.
Therefore, proceeding with ordinal encoding instead of one-hot encoding

"""
# ver3 05/28 Dropping the '도로명' column and proceeding to drop the unprocessed '도로조건 null data

df = df.dropna(subset=["road_condition"])
encoder = OrdinalEncoder(
    categories=[
        [
            "8m미만",
            "12m미만",
            "25m미만",
            "25m이상",
        ]
    ]
)
df["road_condition"] = encoder.fit_transform(df[["road_condition"]])


df_district = df["district"]
df = df.drop(columns=["district"])
df = pd.concat([df, df_district], axis=1)

# 05/28 Extracting district from "시군구", then proceeding with one-hot encoding.
df["district"] = df["district"].apply(lambda x: x.split()[-1])
encoded_district = pd.get_dummies(df["district"])
df = pd.concat([df, encoded_district], axis=1)
df = df.drop(columns=["district"])

df_js = df[df["lease_type"] == "전세"]
df_js_target = df_js["deposit"]

df_js = df_js.drop(columns=["deposit", "monthly_rent_bill", "lease_type"])
df_js = pd.concat([df_js, df_js_target], axis=1)

df_ws = df[df["lease_type"] == "월세"]
df_ws_target = df_ws["monthly_rent_bill"]

df_ws = df_ws.drop(columns=["lease_type", "monthly_rent_bill"])
df_ws = pd.concat([df_ws, df_ws_target], axis=1)

print(df_js.head())
print(df_ws.head())

df_js.to_csv("jeonse_dataset.csv", index=False, encoding="utf-8-sig")
df_ws.to_csv("wolse_dataset.csv", index=False, encoding="utf-8-sig")
