import time
import argparse

from loguru import logger
import pandas as pd

from utils import csvIO, modelIO
from preprocessor import (
    sales_feature,
    data_operation,
    data_integrator,
    data_cleaner,
    data_normalizer,
)
from predictor import DTR, XGBoost, kMeans

def main_old():

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

def main():

#-----------------------------------------------------------------------------------
# Phase 1. Read Data
    # a. Read original training data
    logger.info("Reading dataset, file name: ./dataset/sales_train.csv")
    original_train_data = csvIO.read_csv_to_pddf("./dataset/sales_train.csv")

    # b. Read item information - including category info
    logger.info("Reading dataset, file name: ./dataset/items.csv")
    item_info = csvIO.read_csv_to_pddf("./dataset/train_monthly_sales.csv")

#-----------------------------------------------------------------------------------
# Phase 2. Integrate Training Data and Do Feature Extraction

    # a. Integrate train data to monthly information, then join category information
    logger.info("Aggregate original training data to monthly sales.")
    agg_sales_train_data = data_integrator.integrate_monthly_sales(original_train_data)

    # b. Join Category Info
    logger.info("Join category information.")
    agg_sales_train_data = data_integrator.join_category_info(agg_sales_train_data, item_info)

    # c. Permutate agg_sales_train_data
    agg_sales_train_data = agg_sales_train_data[
        ['date_block_num', 'shop_id', 'item_id', 'item_category_id',
         'avg_sales_price', 'total_sales', 'total_record_count']
    ]

    # d. Do statistics - Monthly sales of specific category on specific shop
    logger.info("Do statistics - Monthly sales of specific category on specific shop")
    category_sales_on_shop_per_month = sales_feature.shop_seasonal_sales_of_category(agg_sales_train_data)

    # e. Do statistics - Monthly sales of all item without distinguishing shop
    logger.info("Do statistics - Monthly sales of all item without distinguishing shop")
    monthly_total_item_sales = sales_feature.item_total_sales(agg_sales_train_data)

    logger.info("Do statistics - Monthly sales of all item category without distinguishing shop")
    monthly_total_item_cat_sales = sales_feature.item_category_total_sales(agg_sales_train_data)

#-----------------------------------------------------------------------------------
# Phase 3. Data Cleaning and Normalization

    # a. Clean outlier of training data
    logger.info("Clean Outliers of Training Data")
    clean_outlier_train_data = data_cleaner.remove_outlier(agg_sales_train_data)

#-----------------------------------------------------------------------------------
# Phase 4. k-Means clustering for extract feature of popularity
    logger.info("Do sales heat auto clustering")
    category_heat_value = kMeans.extract_sales_heat(category_sales_on_shop_per_month)
    csvIO.write_pd_to_csv(
        category_heat_value,
        "category_heat_value.csv",
        False,
    )
    # category_heat_value = csvIO.read_csv_to_pddf(
    #     "./output/category_heat_value.csv"
    # )
    # print(category_heat_value)

    monthly_item_heat_value = kMeans.extract_monthly_item_sales_heat(
        monthly_total_item_sales
    )
    csvIO.write_pd_to_csv(
        monthly_item_heat_value,
        "monthly_item_heat_value.csv",
        False
    )
    # monthly_item_heat_value = csvIO.read_csv_to_pddf(
    #     "./output/monthly_item_heat_value.csv"
    # )

    monthly_item_cat_heat_value = kMeans.extract_monthly_item_cat_sales_heat(
        monthly_total_item_cat_sales
    )
    csvIO.write_pd_to_csv(
        monthly_item_cat_heat_value,
        "monthly_item_cat_heat_value.csv",
        False
    )
    # monthly_item_heat_value = csvIO.read_csv_to_pddf(
    #     "./output/monthly_item_heat_value.csv"
    # )

#-----------------------------------------------------------------------------------
# Phase 5. Do Input Data Normalization

    # Step 1. Integrate features
    logger.info("Join sales heat feature")
    train_data_with_feature = data_integrator.feature_join(
        clean_outlier_train_data,
        category_heat_value,
        monthly_item_heat_value,
        monthly_item_cat_heat_value,
    )
    assert len(clean_outlier_train_data) == len(train_data_with_feature), "Inconsistant length of data before/after join operation"

    train_data_with_feature.drop(
        columns=[
            'cat_shop_total_sales_sum',
            'monthly_total_item_sales_sum',
            'monthly_total_item_cat_sales_sum',
            'cat_shop_total_sales_mean',
            'monthly_total_item_sales_mean',
            'monthly_total_item_cat_sales_mean',
            # 'avg_sales_price',
            'total_record_count',
        ],
        inplace=True
    )

    # Step 2. Normalization
    # logger.info("Data Normalization")
    norm_train_data = (
        data_normalizer.train_norm(train_data_with_feature)
    )
    logger.debug("Use Original value on fitting target - total_sales")

    # Cancel Normalization on Output Data and Specific Columns
    norm_train_data["date_block_num"] = train_data_with_feature["date_block_num"]
    norm_train_data["total_sales"] = train_data_with_feature["total_sales"]
    # norm_train_data = train_data_with_feature
    print(norm_train_data.columns)
    # ts_train_data_norm = ts_train_data

#-----------------------------------------------------------------------------------
# Phase 6. Encode Time Series Data

    # Step 1. Encode all features as time series data
    logger.info("Encode Time Series Data")
    ts_train_data = data_integrator.encode_time_series_data(
        norm_train_data,
        24,
    )

    # Step 2. Do normalization for month ID
    ts_train_data["date_block_num"] = ts_train_data["date_block_num"].apply(
        lambda x: (x % 12) - 0 / 11
    )

    print(len(ts_train_data))

#-----------------------------------------------------------------------------------
# Phase 7. XGBoostRegressor Training and Evaluate Performance
    logger.info("Fit XGBoost Regressor")
    xgbr = XGBoost.train(ts_train_data, 0.20)
    modelIO.save(xgbr, "xgbr_new_feature")
#-----------------------------------------------------------------------------------
# Phase 8. Do Inference Based on Training Result

if __name__ == "__main__":
    main()
