from loguru import logger
import numpy as np
import pandas as pd

from utils import csvIO, modelIO
from preprocessor import sales_feature, data_operation
from predictor import DTR, XGBoost

if __name__ == "__main__":

    # load testing csv file
    inference_data = csvIO.read_csv_to_pddf("./dataset/test.csv")
    print(inference_data)

    # load sales information file
    sales_information = csvIO.read_csv_to_pddf("./output/train_monthly_sales.csv")

    # Query category number (0 ~ 83) of an item by index
    item_idx = csvIO.read_csv_to_pddf("./dataset/items.csv")
    category_of_item = item_idx["item_category_id"]

    # input data - join category of item
    inference_data = inference_data.join(category_of_item, on="item_id")

    # input data - edit month_ID to 2015-11
    inference_data["date_block_num"] = inference_data["shop_id"].apply(lambda x: 34)
    inference_data["month_id"] = inference_data["shop_id"].apply(lambda x: 11)

    # Query Time Series Feature
    inference_ts_data = data_operation.inference_ts_data_join(inference_data, sales_information)

    # load model
    # dtf_8 = modelIO.load("./output/dtr_8.model")
    xgbr = modelIO.load("./output/xgbRegressor_all_ts.model")

    # do inference with input data
    # result = DTR.inference(dtf_8, inference_data)
    result = XGBoost.inference(xgbr, inference_ts_data)

    opres = pd.DataFrame(result, columns=["item_cnt_month"]).reset_index().rename(columns={"index": "ID"})
    opres.loc[opres["item_cnt_month"] < 0,"item_cnt_month"] = 0

    # Denormalization and do rounding
    power_round = lambda x: np.round(np.power(x, 10))
    opres["item_cnt_month"] = opres["item_cnt_month"].apply(power_round)
    print(opres)
    csvIO.write_pd_to_csv(opres, "submission.csv", False)
