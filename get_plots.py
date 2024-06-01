import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor


def feature_importance(dataframe):

    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    model = ExtraTreesRegressor()
    model.fit(X, y)

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    feat_importances.nlargest(10).plot(kind="barh")


def corr_heatmap(dataframe):

    corrmat = dataframe.corr()
    plt.figure(figsize=(10, 10))
    plt.title("Correlation Matrix")
    sns.heatmap(corrmat, annot=True, cmap="RdYlGn")


def get_plot(dataset, distrcit_score):
    dataframe = pd.read_csv(dataset)

    feature_importance(dataframe)

    if (distrcit_score):
        corr_heatmap(dataframe)

    plt.show()


get_plot('jeonse_dataset.csv', False)
get_plot('jeonse_dataset_district_score.csv', True)
get_plot('wolse_dataset.csv', False)
get_plot('wolse_dataset_district_score.csv', True)
