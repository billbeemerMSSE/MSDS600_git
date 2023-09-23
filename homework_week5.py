"""
Homework Week 5
MSDS 600
Bill Beemer
"""

import pandas as pd
from pycaret.classification import predict_model, load_model


def load_data(path):
    """
    Loads churn data into a data frame

    @param path: str. file path str to csv file
    @return: dataframe. dataframe object
    """
    df = pd.read_csv(path, index_col="customerID")
    return df


def make_predictions(dataframe):
    """
    Uses pycaret to make predictions on dataframe

    @param dataframe: dataframe object. 
    @return: predictions. predictions object
    """

    model = load_model("LR_model")
    predictions = predict_model(model, data=dataframe)
    predictions.rename({"prediction_label": "Churn_prediction"}, axis=1, inplace=True)
    predictions["Churn_prediction"].replace({1: "Churn", 0: "No Churn"}, inplace=True)
    return predictions["Churn_prediction"]


if __name__ == "__main__":
    dataframe = load_data("data/new_churn_data.csv")
    predictions = make_predictions(dataframe)
    print("Predictions:")
    print(predictions)