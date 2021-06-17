import time

from loguru import logger
import pandas as pd

from utils import csvIO, modelIO
from preprocessor import sales_feature, data_operation
from predictor import DTR, XGBoost

def main():

    total_month_count = 34
    item_count = 22170
    shop_count = 60

    logger.info("Reading dataset, file name: ./dataset/train_monthly_sales.csv")
    monthly_sales_train = csvIO.read_csv_to_pddf("./dataset/train_monthly_sales.csv")
    # print(monthly_sales_train)
    logger.debug(f"Expected: {total_month_count * item_count * shop_count}, Real: {len(monthly_sales_train)} ({len(monthly_sales_train) * 100 / (total_month_count * item_count * shop_count):.2f}%)")

    # # Do feature extraction (Deprecated)
    # logger.info("Do feature extraction")
    # feature_collection = data_operation.feature_extract(monthly_sales_train)

    # # Encode Data - Input with Item ID, Shop ID and Month ID (Deprecated)
    # logger.info("[Data Composing] Build Data for model fitting")
    # data_in, data_out = data_operation.compose_data(monthly_sales_train.query("item_cnt_day > 0"), feature_collection)  # Filter sales < 0

    # Do Time Series Data Encoding
    logger.info("[Time Series Data] Encode Data for model fitting")
    data_in, data_out = data_operation.encode_time_series_data(monthly_sales_train)

    # decision_tree_model = DTR.train(data_in, data_out)
    # modelIO.save(decision_tree_model, "dtr_8_all_ts")

    xgbr, xgbr_rmse = XGBoost.train(data_in, data_out.drop("shop_id", axis=1))
    modelIO.save(xgbr, f"xgbRegressor_all_ts")

    # xgbr_list = []
    # xgbr_rmse_list = []
    # shop_list = range(60) # 6, 9, 12, 15, 18, 20, 22, 25, 27, 28, 31, 42, 43, 54, 55, 57
    # for i in shop_list:
    #     logger.info(f"Individual shop ID: {i}")
    #     train_in = data_in.query(f"shop_id == {i}").reset_index(drop=True)
    #     train_out = data_out.query(f"shop_id == {i}").reset_index(drop=True)

    #     train_in = train_in.drop("shop_id", axis=1)
    #     train_out = train_out.drop("shop_id", axis=1)

    #     print(train_in)
    #     print(train_out)

    #     # decision_tree_model = DTR.train(data_in, data_out)
    #     # modelIO.save(decision_tree_model, "dtr_8")

    #     xgbr, xgbr_rmse = XGBoost.train(train_in, train_out)
    #     xgbr_list.append(xgbr)
    #     xgbr_rmse_list.append(xgbr_rmse)

    # list_avg = lambda x: sum(x) / len(x)
    # logger.info(f"Overall RMSE average: {list_avg(xgbr_rmse_list)}")
    # print(xgbr_rmse_list)

    # for i in range(len(xgbr_list)):
    #     modelIO.save(xgbr_list[i], f"xgbRegressor_shop_{i}")

if __name__ == "__main__":
    main()
