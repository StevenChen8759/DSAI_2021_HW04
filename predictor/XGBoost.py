import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from loguru import logger

from utils import csvIO
from preprocessor import data_operation

def train(
    data_X: pd.DataFrame,
    data_Y: pd.DataFrame,
    test_ratio: float = 0.25,
):
    logger.info("Fit XGBoost Regressor")

    test_X = data_X.sample(frac=test_ratio, replace=False).fillna(0)
    test_Y = data_Y.iloc[test_X.index].fillna(0)

    test_df_index = data_X.index.isin(test_X.index)

    train_X = data_X.loc[~test_df_index].fillna(0)
    train_Y = data_Y.loc[~test_df_index].fillna(0)

    xgbr = XGBRegressor(
        n_estimators=1000,         # Maximum Epoches
        eta=0.1,
        max_depth=6,
        subsample=0.75,
        colsample_bytree=0.75
    )
    # 0.1, 6, 0.75, 1.0 -> 0.60
    # 0.1, 6, 0.75, 0.8 -> 0.71
    # 0.1, 6, 0.70, 0.95 -> 0.81 (59)
    # 0.1, 6, 0.70, 0.75 -> 0.78 (69)
    # 0.1, 6, 0.65, 0.75 -> 1.05 (57)
    # 0.1, 6, 0.65, 0.80 -> 1.81 (215)
    # 0.1, 6, 0.65, 0.70 -> 0.80 (69) / 1.61 (134) / 4.14 (173) / 0.57 (91)

    xgbr.fit(
        train_X,
        train_Y,
        early_stopping_rounds=20,
        eval_metric = "rmse",
        eval_set = [[test_X, test_Y]],
        verbose=True
    )

    predict_Y = np.round(xgbr.predict(test_X))

    model_rmse = mean_squared_error(test_Y, predict_Y, squared=False)

    logger.debug(f"Prediction RMSE: {model_rmse:.5f}")

    return xgbr, model_rmse

def inference(model, inference_data: pd.DataFrame):


    # Do statistical feature extraction (Deprecated)
    # feature_collection = data_operation.feature_extract(sales_information)

    # query and join feature for model input (Deprecated)
    # inf_data = data_operation.feature_join(
    #     inference_data,
    #     feature_collection,
    # )

    # Join time series features

    return model.predict(inference_data)
