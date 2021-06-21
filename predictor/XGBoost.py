import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from loguru import logger

from utils import csvIO
from preprocessor import data_operation

def train_old(
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
        eta=0.05,
        max_depth=8,
        subsample=0.75,
        colsample_bytree=0.85
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


def train(
    input_data: pd.DataFrame,
    test_ratio: float = 0.25,
):

    # Do data shuffling
    input_data_shuffled = (
        input_data.sample(frac=1.0, replace=False)
                  .reset_index(drop=True)
    )
    assert len(input_data_shuffled) == len(input_data), "Inconsistant length between shuffled and non-shuggled data"

    # Split Training Set and Testing Set
    logger.debug(f"Split Testing Set, Ratio: {test_ratio:.2f}")
    train_data = (
        input_data_shuffled.sample(
            frac=(1 - test_ratio),
            replace=False
        )
    )
    test_data = input_data_shuffled[~input_data.index.isin(train_data.index)]
    assert len(train_data) + len(test_data) == len(input_data_shuffled), "Incorect length of sum of training and testing set."

    # Split input and output
    logger.debug("Split Input and Output")
    train_X = train_data.drop(
        columns=[
            'avg_sales_price',
            'total_sales',
            # 'total_record_count',
            # 'cat_shop_total_sales_mean',
            'cat_shop_sales_heat',
            # 'monthly_total_item_sales_mean',
            # 'monthly_total_item_cat_sales_mean',
            'monthly_total_item_sales_heat',
            # 'cat_shop_total_sales_sum',
            # 'monthly_total_item_sales_sum',
            # 'monthly_total_item_cat_sales_sum',
            'monthly_total_item_cat_sales_heat',
        ]
    )
    train_Y = train_data[
        ['total_sales']
    ]
    test_X = test_data.drop(
        columns=[
            'avg_sales_price',
            'total_sales',
            # 'total_record_count',
            # 'cat_shop_total_sales_mean',
            'cat_shop_sales_heat',
            # 'monthly_total_item_sales_mean',
            # 'monthly_total_item_cat_sales_mean',
            'monthly_total_item_sales_heat',
            # 'cat_shop_total_sales_sum',
            # 'monthly_total_item_sales_sum',
            # 'monthly_total_item_cat_sales_sum',
            'monthly_total_item_cat_sales_heat',
        ]
    )
    test_Y = test_data[
        ['total_sales',]
    ]
    print("Feature count: %d" % len(train_X.columns))
    print(train_X.columns)

    # Declare XGBoost Regressor
    xgbr = XGBRegressor(
        n_estimators=1200,         # Maximum Epoches
        eta=0.1,
        max_depth=10,
        subsample=0.75,
        colsample_bytree=0.75,
    )

    # Make Pipeline with Min Max Scalar
    # xgbr_with_norm = Pipeline(
    #     [
    #         ('normalizer', MinMaxScaler()),
    #         ('xgbregressor', xgbr),
    #     ]
    # )


    # Fit XGBoost Regressor with Pipeline
    # xgbr_with_norm.fit(
    #     train_X.values,
    #     train_Y.values,
    #     xgbregressor__early_stopping_rounds=20,
    #     xgbregressor__eval_metric="rmse",
    #     xgbregressor__eval_set=[
    #         [train_X.values, train_Y.values.ravel()],
    #         [test_X.values, test_Y.values.ravel()]
    #     ],
    #     xgbregressor__verbose=True,
    # )

    # Fit XGBoost Regressor
    xgbr.fit(
        train_X,
        train_Y,
        early_stopping_rounds=20,
        eval_metric = "rmse",
        eval_set = [[train_X, train_Y], [test_X, test_Y]],
        verbose=True
    )

    predict_Y = xgbr.predict(test_X)

    model_rmse = mean_squared_error(test_Y, predict_Y, squared=False)

    logger.debug(f"Prediction RMSE: {model_rmse:.5f}")

    return xgbr


def inference(model, inference_data: pd.DataFrame):


    # Do statistical feature extraction (Deprecated)
    # feature_collection = data_operation.feature_extract(sales_information)

    # query and join feature for model input (Deprecated)
    # inf_data = data_operation.feature_join(
    #     inference_data,
    #     feature_collection,
    # )

    # Update date_block_num to month ID
    inference_data["date_block_num"] = inference_data["date_block_num"].apply(lambda x: (x % 12) + 1)

    return model.predict(inference_data)
