from sklearn.preprocessing import StandardScaler
import pandas as pd


def district_only(df, col_to_drop, target):
    X = df.iloc[:, 0:col_to_drop]

    df_only_distrcit = df.drop(X.columns, axis=1)

    district_corr = df_only_distrcit.corr()[target].drop(target)
    district_index = district_corr.index  # 지역명 인덱스 저장

    district_corr = district_corr.values.reshape(-1, 1)
    district_scores = StandardScaler().fit_transform(district_corr)

    # column : 동, row: district score
    district_scores_df = pd.DataFrame(
        district_scores, index=district_index, columns=["district_score"]
    ).T

    return district_index, district_scores_df


def get_district_score(row, district_index, district_scores_df):
    for col in district_index:
        if row[col] == 1:
            return district_scores_df[col].values[0]
    return 0  # 모든 값이 False인 경우


def district_score_dataset(dataset, col_to_drop, target):
    df = pd.read_csv(dataset)
    pd.set_option("display.max_seq_items", None)

    district_index, district_scores_df = district_only(df, col_to_drop, target)

    df["district_score"] = df.apply(lambda row: get_district_score(
        row, district_index, district_scores_df), axis=1)

    df = df.drop(columns=district_index)
    target_col = df.pop(target)
    df[target] = target_col

    return df


df = district_score_dataset("data/jeonse_dataset.csv", 4, "deposit")


# normalize된 전세 데이터를 csv로 저장
df.to_csv(
    "data/eonse_dataset_district_score.csv",
    mode="w",
    index=False,
    encoding="utf-8-sig",
)

df = district_score_dataset("data/wolse_dataset.csv", 5, "monthly_rent_bill")


# normalize된 전세 데이터를 csv로 저장
df.to_csv(
    "data/wolse_dataset_district_score.csv",
    mode="w",
    index=False,
    encoding="utf-8-sig",
)
