import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor


def feature_importance(dataset, dataframe):

    X = dataframe.iloc[:, :-1]
    y = dataframe.iloc[:, -1]

    model = ExtraTreesRegressor()
    model.fit(X, y)

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    plt.figure(figsize=(10, 6))
    plt.title(dataset + " Feature Importance")
    feat_importances.nlargest(10).plot(kind="barh")


def corr_heatmap(dataset, dataframe):

    corrmat = dataframe.corr()
    plt.figure(figsize=(15, 10))
    plt.title(dataset + " Correlation Matrix")
    sns.heatmap(corrmat, annot=True, cmap="RdYlGn")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)


def get_plot(dataset, distrcit_score):
    dataframe = pd.read_csv(dataset)

    feature_importance(dataset, dataframe)

    if (distrcit_score):
        corr_heatmap(dataset, dataframe)

    plt.show()


# get_plot('data/wolse_dataset.csv', False)
get_plot('data/wolse_dataset_district_score.csv', True)

# get_plot('data/jeonse_dataset.csv', False)
# get_plot('data/jeonse_dataset_district_score.csv', True)
