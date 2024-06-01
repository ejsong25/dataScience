from sklearn.preprocessing import StandardScaler
import pandas as pd

# Function to calculate district scores based on correlation with the target variable


def district_only(df, col_to_drop, target):
    # Select the initial columns to drop
    X = df.iloc[:, 0:col_to_drop]

    # Drop the selected columns from the dataframe
    df_only_distrcit = df.drop(X.columns, axis=1)

    # Calculate the correlation of the remaining columns with the target variable and drop the target itself
    district_corr = df_only_distrcit.corr()[target].drop(target)
    district_index = district_corr.index  # Save the index of the district names

    # Reshape the correlation values and standardize them
    district_corr = district_corr.values.reshape(-1, 1)
    district_scores = StandardScaler().fit_transform(district_corr)

    # Create a dataframe of the standardized district scores
    district_scores_df = pd.DataFrame(
        district_scores, index=district_index, columns=["district_score"]
    ).T

    return district_index, district_scores_df

# Function to get the district score for a specific row


def get_district_score(row, district_index, district_scores_df):
    # Iterate through the district columns to find the district score
    for col in district_index:
        if row[col] == 1:
            return district_scores_df[col].values[0]
    return 0  # Return 0 if all values are False

# Function to create a dataset with district scores


def district_score_dataset(dataset, col_to_drop, target):
    # Read the dataset
    df = pd.read_csv(dataset)
    pd.set_option("display.max_seq_items", None)

    # Get the district index and scores dataframe
    district_index, district_scores_df = district_only(df, col_to_drop, target)

    # Apply the district scores to the dataframe
    df["district_score"] = df.apply(lambda row: get_district_score(
        row, district_index, district_scores_df), axis=1)

    # Drop the district columns and move the target column to the end
    df = df.drop(columns=district_index)
    target_col = df.pop(target)
    df[target] = target_col

    return df


# Process the jeonse dataset and save the results
df = district_score_dataset("data/jeonse_dataset.csv", 4, "deposit")

# Save the normalized jeonse dataset to a CSV file
df.to_csv(
    "data/jeonse_dataset_district_score.csv",
    mode="w",
    index=False,
    encoding="utf-8-sig",
)

# Process the wolse dataset and save the results
df = district_score_dataset("data/wolse_dataset.csv", 5, "monthly_rent_bill")

# Save the normalized wolse dataset to a CSV file
df.to_csv(
    "data/wolse_dataset_district_score.csv",
    mode="w",
    index=False,
    encoding="utf-8-sig",
)
