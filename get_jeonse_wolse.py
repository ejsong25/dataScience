from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from datetime import datetime
import pandas as pd
import numpy as np

"""
Replace DirtyData into null
"""
df = pd.read_csv("raw_data/seongnam_dataset.csv", na_values="-")
pd.set_option("display.max_seq_items", None)

"""
Change feature names to English
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

# Rename columns
df = df.rename(columns=column_mapping)

"""
Basic statistics data
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
print("[Before processing (isnull().sum()): seongname_dataset.csv]\n")
print(df.isnull().sum())
print(
    "\n====================================================================================================================\n"
)

"""
[Extract meaningful data]
Drop columns - lot_number, contract_year_month, contract_day, road_name, contract_type, and all columns after that
Use columns - district, road_condition, contract_area_m2, lease_type, deposit (10,000 KRW), monthly_rent_bill (10,000 KRW), construction_year, road_name, contract_period
"""

"""
ver2. 05/24

After the second preprocessing meeting, it was decided not to use the district for now.
We will use it again when further developing for trends and evaluations by neighborhood later.

ver3. 05/28 Added district column
"""

# ver4. 05/30 Drop contract_year_month - unnecessary for price prediction, will use for future statistical data extraction
indices_to_use = [0, 2, 3, 4, 7, 8, 9, 11]
df = df.iloc[:, indices_to_use]

print("[Sample data after dropping useless columns]\n")
print(df.head(10))
print(
    "\n====================================================================================================================\n"
)

"""
Remove commas in deposit (10,000 KRW)
"""
df["deposit"] = df["deposit"].fillna("0").apply(
    lambda x: int(x.replace(",", "")))

"""
Incorrect data in lease_type:

deposit == 0
lease_type is monthly rent but monthly_rent_bill == 0
lease_type is jeonse but monthly_rent_bill != 0

Remove data
"""
index_to_drop = df[(df["deposit"] < 300)].index
df.drop(index_to_drop, inplace=True)

index_to_drop = df[(df["lease_type"] == "월세") & (
    df["monthly_rent_bill"] == 0)].index
df.drop(index_to_drop, inplace=True)

index_to_drop = df[(df["lease_type"] == "전세") & (
    df["monthly_rent_bill"] != 0)].index
df.drop(index_to_drop, inplace=True)

"""
Building age = current year - construction_year
The smaller the number, the better
"""
df = df.dropna(subset=["construction_year"])
df["building_age"] = (datetime.now().year -
                      df["construction_year"]).astype(int)
df = df.drop(columns=["construction_year"])

"""
Contract period: 202401 ~ 202601
1. Extract two years based on '~' (2024, 2026)
2. Subtract the start year from the end year
3. Fill null values with the average value
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

"""
Road condition: ['less than 8m', 'less than 12m', 'less than 25m', 'more than 25m']: the width of the road adjacent to the property

The wider the road, the better the traffic.
Therefore, use ordinal encoding instead of one-hot encoding
"""
# ver3 05/28 Drop 'road_condition' null data while dropping the road_name column

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

# 05/28 Extract dong from district, perform one-hot encoding
df["district"] = df["district"].apply(lambda x: x.split()[-1])
encoded_district = pd.get_dummies(df["district"])
df = pd.concat([df, encoded_district], axis=1)
df = df.drop(columns=["district"])


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

'''
divided jeonse, wolse data
'''
df_js = df[df["lease_type"] == "전세"]
df_js_target = df_js["deposit"]

df_js = df_js.drop(columns=["deposit", "monthly_rent_bill", "lease_type"])
df_js = pd.concat([df_js, df_js_target], axis=1)

print("[df_js.describe()]\n")
print(df_js.describe())
print(
    "\n====================================================================================================================\n"
)

min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

# Apply Min-Max scaling to contract_area_m2, contract_period
df_js['contract_area_m2'] = min_max_scaler.fit_transform(
    df_js[['contract_area_m2']])
df_js['contract_period'] = min_max_scaler.fit_transform(
    df_js[['contract_period']])

# Apply standardization to building_age
df_js['building_age'] = standard_scaler.fit_transform(df_js[['building_age']])

df_ws = df[df["lease_type"] == "월세"]
df_ws_target = df_ws["monthly_rent_bill"]

df_ws = df_ws.drop(columns=["lease_type", "monthly_rent_bill"])
df_ws = pd.concat([df_ws, df_ws_target], axis=1)

print("[df_ws.describe()]\n")
print(df_ws.describe())
print(
    "\n====================================================================================================================\n"
)

# Apply Min-Max scaling to road_condition, contract_area_m2, contract_period
df_ws['contract_area_m2'] = min_max_scaler.fit_transform(
    df_ws[['contract_area_m2']])
df_ws['contract_period'] = min_max_scaler.fit_transform(
    df_ws[['contract_period']])

# Apply standardization to building_age
df_ws['building_age'] = standard_scaler.fit_transform(df_ws[['building_age']])

# Apply log transformation and then standardization to deposit
df_ws['deposit'] = np.log1p(df_ws['deposit'])  # Log transformation
df_ws['deposit'] = standard_scaler.fit_transform(
    df_ws[['deposit']])  # Standardization

df_js.to_csv("data/jeonse_dataset.csv",
             index=False, encoding="utf-8-sig")
df_ws.to_csv("data/wolse_dataset.csv", index=False, encoding="utf-8-sig")

print("[After processing (isnull().sum()): jeonse_dataset.csv]\n")
print(df_js.isnull().sum())
print(
    "\n====================================================================================================================\n"
)

print("[After processing (isnull().sum()): wolse_dataset.csv]\n")
print(df_ws.isnull().sum())
print(
    "\n====================================================================================================================\n"
)

print("[After Scaling df_js.describe()]\n")
print(df_js.describe())
print(
    "\n====================================================================================================================\n"
)

print("[After Scaling df_ws.describe()]\n")
print(df_ws.describe())
print(
    "\n====================================================================================================================\n"
)
