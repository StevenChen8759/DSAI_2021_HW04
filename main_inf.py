from loguru import logger
import numpy as np
import pandas as pd

from utils import csvIO, modelIO
from preprocessor import (
    sales_feature,
    data_operation,
    data_integrator,
    data_cleaner,
    data_normalizer,
)
from predictor import DTR, XGBoost

def main_old():
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


def main():
#-----------------------------------------------------------------------------------
# Phase 1. Read All Necessary File and Model

    logger.info("Load Files")
    # Load testing csv file
    inference_data = csvIO.read_csv_to_pddf("./dataset/test.csv")

    # Load sales information file
    sales_information = csvIO.read_csv_to_pddf("./dataset/sales_train.csv")

    # Query category number (0 ~ 83) of an item by index
    item_idx = csvIO.read_csv_to_pddf("./dataset/items.csv")

    # Load category heat value info
    category_heat_value = csvIO.read_csv_to_pddf("./output/category_heat_value.csv")

    # Load category heat value info
    monthly_item_heat_value = csvIO.read_csv_to_pddf("./output/monthly_item_heat_value.csv")

    monthly_item_cat_heat_value = csvIO.read_csv_to_pddf("./output/monthly_item_cat_heat_value.csv")

    # Load XBGRegressor
    xgbr = modelIO.load("./output/xgbr_new_feature.model")

#-----------------------------------------------------------------------------------
# Phase 2. Generate Statistic Info and Join Necessary Information
    # Sales Info - Generate Statistic Info, then join with catgory info
    logger.info("Do data statistics and join with item category")
    stat_sales_data = data_integrator.integrate_monthly_sales(sales_information)
    stat_sales_data_with_cat = (
        data_integrator.join_category_info(
            stat_sales_data,
            item_idx
        )
    )

    # Sales Info - Permuate input order
    stat_sales_data_with_cat = stat_sales_data_with_cat[
        ['date_block_num', 'shop_id', 'item_id', 'item_category_id',
         'avg_sales_price', 'total_sales',  'total_record_count']
    ]

    # Sales Info - Join With Sales Heat
    logger.info("Join data statistics with sales heat")
    stat_sales_info_with_feature = data_integrator.feature_join(
        stat_sales_data_with_cat,
        category_heat_value,
        monthly_item_heat_value,
        monthly_item_cat_heat_value,
    )

    stat_sales_info_with_feature.drop(
        columns=[
            'cat_shop_total_sales_mean',
            'monthly_total_item_sales_mean',
            'monthly_total_item_cat_sales_mean',
            # 'avg_sales_price',
            'total_record_count',
            'cat_shop_total_sales_sum',
            'monthly_total_item_sales_sum',
            'monthly_total_item_cat_sales_sum',
        ],
        inplace=True
    )

    # Inference data - join category of item
    logger.info("Join inference data with item category")
    category_of_item = item_idx["item_category_id"]
    inference_data_with_cat = inference_data.join(category_of_item, on="item_id")
    # print(inference_data_with_cat)

#-----------------------------------------------------------------------------------
# Phase 3. Do Normalization on inference data

    # logger.info("Do Statistical Data Normalization")
    stat_sales_info_norm = data_normalizer.train_norm(stat_sales_info_with_feature)
    stat_sales_info_norm["date_block_num"] = stat_sales_info_with_feature["date_block_num"]
    # stat_sales_info_norm = stat_sales_info_with_feature

    # logger.info("Do Inference Data Normalization")
    inference_data_norm = data_normalizer.train_norm(inference_data_with_cat)
    inference_data_norm["ID"] = inference_data_with_cat["ID"]
    # inference_data_norm = inference_data_with_cat

#-----------------------------------------------------------------------------------
# Phase 3. Encode Time Series Data For Inference

    logger.info("Encode Time Series Data for inference")
    inference_ts_data = (
        data_integrator.make_inference_ts_data(
            inference_data_norm,
            stat_sales_info_norm,
            24,
        )
    )

    # Do normalization for month ID
    inference_ts_data["date_block_num"] = inference_ts_data["date_block_num"].apply(
        lambda x: (x % 12) - 0 / 11
    )

    # Drop ID and date_block_num for inference
    inference_ts_data.drop(
        columns=[
            "ID",
            # "date_block_num",
        ],
        inplace=True,
    )
    # inference_ts_data["total_sales"] = inference_data_with_cat["total_sales"]
    print(len(inference_ts_data.columns))
    print(inference_ts_data.columns)
    print(inference_ts_data)

#-----------------------------------------------------------------------------------
# Phase 5. Do XGBoost Inference

    logger.info("Do XGBoost Inference")
    result = XGBoost.inference(xgbr, inference_ts_data)

#-----------------------------------------------------------------------------------
# Phase 6. Output Inference Result

    opres = pd.DataFrame(result, columns=["item_cnt_month"]).reset_index().rename(columns={"index": "ID"})
    opres.loc[opres["item_cnt_month"] < 0,"item_cnt_month"] = 0

    print(opres)
    csvIO.write_pd_to_csv(opres, "submission.csv", False)

if __name__ == "__main__":
    main()
