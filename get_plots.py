import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor

# Function to plot feature importance using Extra Trees Regressor


def feature_importance(dataset, dataframe):
    # Separate features (X) and target (y) from the dataframe
    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    # Initialize and fit the model
    model = ExtraTreesRegressor()
    model.fit(X, y)

    # Get feature importances
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    # Plot the top 10 feature importances
    plt.figure(figsize=(10, 6))
    plt.title(dataset + " Feature Importance")
    feat_importances.nlargest(10).plot(kind="barh")

# Function to plot the correlation heatmap


def corr_heatmap(dataset, dataframe):
    # Calculate the correlation matrix
    corrmat = dataframe.corr()

    # Plot the heatmap
    plt.figure(figsize=(15, 10))
    plt.title(dataset + " Correlation Matrix")
    sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.yticks(rotation=45)  # Rotate y-axis labels

# Function to generate plots


def get_plot(dataset, district_score):
    # Read the dataset
    dataframe = pd.read_csv(dataset)

    # Plot feature importance
    feature_importance(dataset, dataframe)

    # If district_score is True, plot the correlation heatmap
    if district_score:
        corr_heatmap(dataset, dataframe)

    # Display the plots
    plt.show()


get_plot('data/wolse_dataset.csv', False)
get_plot('data/wolse_dataset_district_score.csv', True)

get_plot('data/jeonse_dataset.csv', False)
get_plot('data/jeonse_dataset_district_score.csv', True)
