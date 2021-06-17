
import pandas as pd
from pandarallel import pandarallel
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from loguru import logger

from utils import csvIO
from preprocessor import data_operation

def train(
    data_X: pd.DataFrame,
    data_Y: pd.DataFrame,
    test_ratio: float = 0.25,
):
    logger.info("Fit Decision Tree Regressor")

    # Do data range mapping
    # logger.debug("Waiting for range shrinking...")
    # pandarallel.initialize()
    # data_X.parallel_apply(lambda x: x ** 2, axis=1)
    # data_Y.parallel_apply(lambda x: x ** 2, axis=1)
    # logger.debug("Done")

    test_X = data_X.sample(frac=test_ratio, replace=False).fillna(0)
    test_Y = data_Y.iloc[test_X.index].fillna(0)

    test_df_index = data_X.index.isin(test_X.index)

    train_X = data_X.loc[~test_df_index].fillna(0)
    train_Y = data_Y.loc[~test_df_index].fillna(0)

    dtr_8 = DecisionTreeRegressor(
        max_depth=4,
        criterion="mse",
        max_features="sqrt",
    )

    dtr_8.fit(train_X, train_Y)
    score = dtr_8.score(test_X, test_Y)

    # cv_score = cross_val_score(dtr_8, test_X, test_Y, scoring="neg_root_mean_squared_error")
    # print(cv_score)

    predict_Y = dtr_8.predict(test_X)
    model_rmse = mean_squared_error(test_Y, predict_Y, squared=False)

    logger.debug(f"Score: {score:.5f}, Prediction RMSE: {model_rmse:.5f}")

    return dtr_8


def inference(model, inference_data: pd.DataFrame):

    # load sales information file and do statistical feature extraction
    sales_information = csvIO.read_csv_to_pddf("./output/train_monthly_sales.csv")
    feature_collection = training_data_operation.feature_extract(sales_information)

    # Query category number (0 ~ 83) of an item by index
    item_idx = csvIO.read_csv_to_pddf("./dataset/items.csv")
    category_of_item = item_idx["item_category_id"]

    # input data - join category of item
    inference_data = inference_data.join(category_of_item, on="item_id")

    # input data - edit month_ID
    # FIXME: for Nov2015 inference custmoize, be aware of this statement
    inference_data["month_id"] = inference_data["shop_id"].apply(lambda x: 11)

    # reorder the column to fit input order
    new_order = ["month_id", "shop_id", "item_id", "item_category_id"]
    inference_data = inference_data[new_order]

    # query and join feature for model input
    inf_data = training_data_operation.feature_join(
        inference_data,
        feature_collection,
    )

    columns_inf = [
        'item_cnt_day',
        # 'item_cnt_day_category_month',
        'item_cnt_day_item_shop',
        # 'item_cnt_day_category_shop'
    ]
    inf_data = inf_data[columns_inf].fillna(0)

    return model.predict(inf_data)
