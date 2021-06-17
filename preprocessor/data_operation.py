from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from preprocessor import sales_feature


def feature_extract(monthly_sales_train: pd.DataFrame):

    feature_collection = {}

    month_count = 12
    total_month_count = 34
    item_count = 22170
    category_count = 84
    shop_count = 60

    logger.info("Statistical average sales with month and item...")
    item_month_avg_sales = sales_feature.evaluate_mean(
        monthly_sales_train,
        ["date_block_num", "item_id"],
        ["item_cnt_day"],
    )
    logger.debug(f"Expected: {month_count * item_count}, Real: {len(item_month_avg_sales)} ({len(item_month_avg_sales) * 100 / (month_count * item_count):.2f}%)")
    # print(item_month_avg_sales.isnull().values.any())
    feature_collection["sales_item_month"] = item_month_avg_sales.fillna(0)

    logger.info("Statistical average sales with month and category of item...")
    category_month_avg_sales = sales_feature.evaluate_mean(
        monthly_sales_train,
        ["date_block_num", "item_category_id"],
        ["item_cnt_day"],
    )
    logger.debug(f"Expected: {month_count * category_count}, Real: {len(category_month_avg_sales)} ({len(category_month_avg_sales) * 100 / (month_count * category_count):.2f}%)")
    # print(category_month_avg_sales.isnull().values.any())
    feature_collection["sales_category_month"] = category_month_avg_sales.fillna(0)

    logger.info("Statistical average sales with shop and item...")
    item_shop_avg_sales = sales_feature.evaluate_mean(
        monthly_sales_train,
        ["shop_id", "item_id"],
        ["item_cnt_day"],
    )
    logger.debug(f"Expected: {shop_count * item_count}, Real: {len(item_shop_avg_sales)} ({len(item_shop_avg_sales) * 100 / (shop_count * item_count):.2f}%)")
    # print(item_shop_avg_sales.isnull().values.any())
    feature_collection["sales_item_shop"] = item_shop_avg_sales.fillna(0)

    logger.info("Statistical average sales with shop and category...")
    category_shop_avg_sales = sales_feature.evaluate_mean(
        monthly_sales_train,
        ["shop_id", "item_category_id"],
        ["item_cnt_day"],
    )
    logger.debug(f"Expected: {shop_count * category_count}, Real: {len(category_shop_avg_sales)} ({len(category_shop_avg_sales) * 100 / (shop_count * category_count):.2f}%)")
    # print(category_shop_avg_sales.isnull().values.any())
    feature_collection["sales_category_shop"] = category_shop_avg_sales.fillna(0)

    return feature_collection

def compose_data(
    data_train: pd.DataFrame,
    feature_collection: Dict["str", pd.DataFrame]
):
    # Copy input data and remove unused columns
    data_compose = data_train[["date_block_num", "month_id", "shop_id", "item_id", "item_category_id", "item_cnt_day"]]

    # print(feature_collection.keys())

    logger.debug("Compose Statistical Features")
    # Join data with extracted feature - item-month average sales
    data_compose = data_compose.merge(
        feature_collection["sales_item_month"],
        how='inner',
        left_on=['month_id', 'item_id'],
        right_on = ['month_id', 'item_id'],
        suffixes = ["", "_item_month"],
    )

    # Join data with extracted feature - category-month average sales
    data_compose = data_compose.merge(
        feature_collection["sales_category_month"],
        how='inner',
        left_on=['month_id', 'item_category_id'],
        right_on = ['month_id', 'item_category_id'],
        suffixes = ["", "_category_month"],
    )

    # Join data with extracted feature - item-shop average sales
    data_compose = data_compose.merge(
        feature_collection["sales_item_shop"],
        how='inner',
        left_on=['shop_id', 'item_id'],
        right_on = ['shop_id', 'item_id'],
        suffixes = ["", "_item_shop"],
    )

    # Join data with extracted feature - category-shop average sales
    data_compose = data_compose.merge(
        feature_collection["sales_category_shop"],
        how='inner',
        left_on=['shop_id', 'item_category_id'],
        right_on = ['shop_id', 'item_category_id'],
        suffixes = ["", "_category_shop"],
    )

    # Filter

    logger.debug("Compose the sales and average price last month")



    columns_in = [
        'month_id',
        'shop_id',
        'item_id',
        'item_category_id',
        'item_cnt_day_item_month',
        'item_cnt_day_category_month',
        'item_cnt_day_item_shop',
        'item_cnt_day_category_shop'
        # 'avg_item_price_weighted',
        # 'item_sales_last_month',
        ]
    data_in = data_compose[columns_in]

    columns_out = [
        'month_id',
        'shop_id',
        'item_id',
        'item_cnt_day',
    ]
    data_out = data_compose[columns_out]

    return data_in, data_out


def feature_join(
    data_inference: pd.DataFrame,
    feature_collection: Dict["str", pd.DataFrame]
):
    data_compose = data_inference

    logger.debug("Compose Statistical Features")
    # Join data with extracted feature - item-month average sales
    data_compose = data_compose.merge(
        feature_collection["sales_item_month"],
        how='left',
        left_on=['month_id', 'item_id'],
        right_on = ['month_id', 'item_id'],
        suffixes = ["", "_item_month"],
    )

    # Join data with extracted feature - category-month average sales
    data_compose = data_compose.merge(
        feature_collection["sales_category_month"],
        how='left',
        left_on=['month_id', 'item_category_id'],
        right_on = ['month_id', 'item_category_id'],
        suffixes = ["", "_category_month"],
    )

    # Join data with extracted feature - item-shop average sales
    data_compose = data_compose.merge(
        feature_collection["sales_item_shop"],
        how='left',
        left_on=['shop_id', 'item_id'],
        right_on = ['shop_id', 'item_id'],
        suffixes = ["", "_item_shop"],
    )

    # Join data with extracted feature - category-shop average sales
    data_compose = data_compose.merge(
        feature_collection["sales_category_shop"],
        how='left',
        left_on=['shop_id', 'item_category_id'],
        right_on = ['shop_id', 'item_category_id'],
        suffixes = ["", "_category_shop"],
    )

    return data_compose


def encode_time_series_data(data_train: pd.DataFrame):

    # Filter unused columns
    data_original: pd.DataFrame = data_train.drop("item_weighted_sales", axis=1)
    logger.debug(f"Original Data Length: {len(data_original)}")

    # Rename columns for better readability
    col_rename_mapper = {
        "item_cnt_day": "item_sales",
        "avg_item_price_weighted": "item_avg_price",
    }
    data_original = data_original.rename(columns=col_rename_mapper)

    # Do Normalization - Remove data with infinite value
    data_original.replace([np.inf, -np.inf], 0, inplace=True)
    logger.debug(f"Replace infintie value with 0, Length: {len(data_original)}")

    # Do Normalization - Filter value smaller than zero
    data_original.query("item_sales >= 0 and item_avg_price > 0", inplace=True)
    logger.debug(f"Remove data with negative value, Length: {len(data_original)}")

    # Do Normalization - Filter sales bigger than a specific value
    # data_original.query("item_sales <= 100", inplace=True)
    # logger.debug(f"Remove outlier of sales, Length: {len(data_original)}")

    # Do Normalization - Filter average price bigger than a specific value
    # data_original.query("item_avg_price <= 10000", inplace=True)
    # logger.debug(f"Remove outlier of sales, Length: {len(data_original)}")

    # Do Normalization - Do range mapping
    data_original["item_sales"] = data_original["item_sales"].apply(np.log10)
    data_original["item_avg_price"] = data_original["item_avg_price"].apply(lambda x: x ** 0.125)

    logger.debug(f"sales data range mapped to ({data_original['item_sales'].min():.2f}, {data_original['item_sales'].max():.2f})")
    logger.debug(f"average price data range mapped to ({data_original['item_avg_price'].min():.2f}, {data_original['item_avg_price'].max():.2f})")
    # distribution_sales = data_original["item_sales"].plot.hist()
    # distribution_sales.figure.savefig("./output/original_data_distributuion_norm_sales.jpg")
    # distribution_sales = data_original["item_avg_price"].plot.hist()
    # distribution_sales.figure.savefig("./output/original_data_distributuion_norm_avg_price.jpg")

    # Filter data from 2013-01 to 2013-06 as original time series data
    data_train_ts = data_original.query("date_block_num >= 6")
    # print(data_train_ts)

    # Join historical sales information
    for i in range(1, 7):
        sales_px = data_original.query(f"date_block_num <= {33 - i}")
        sales_px["date_block_num"].apply(lambda x: x + i)   # shift date_block_num

        data_train_ts = data_train_ts.merge(
            sales_px,
            how="left",
            left_on=['date_block_num', 'shop_id', 'item_id', 'item_category_id'],
            right_on = ['date_block_num','shop_id', 'item_id', 'item_category_id'],
            suffixes=["", f"_p{i}"]
        )

    # Replace NaN with 0
    data_train_ts = data_train_ts.fillna(0)

    # Do data shuffling
    data_train_ts = data_train_ts.sample(frac=1).reset_index(drop=True)
    data_train_ts["month_id"] = data_train_ts["date_block_num"].apply(lambda x: (x % 12) + 1)

    # Compose Input Data
    column_in = [
        # 'date_block_num',
        'month_id',             # Month ID of P0
        'shop_id',
        'item_id',
        'item_category_id',
        # 'item_sales'
        # 'item_avg_price',
        'item_sales_p1', 'item_avg_price_p1',
        'item_sales_p2', 'item_avg_price_p2',
        'item_sales_p3', 'item_avg_price_p3',
        'item_sales_p4', 'item_avg_price_p4',
        'item_sales_p5', 'item_avg_price_p5',
        'item_sales_p6', 'item_avg_price_p6'
    ]
    data_in = data_train_ts[column_in]

    # Compose Output data
    column_out = [
        # 'date_block_num',
        'shop_id',
        # 'item_id',
        # 'item_category_id',
        'item_sales'
        # 'item_avg_price',
    ]
    data_out = data_train_ts[column_out]

    # print(data_in)
    # print(data_in.columns.values)
    # print(data_out)
    # assert False, "Unit Test Assertion"
    return data_in, data_out


def inference_ts_data_join(
    data_inference: pd.DataFrame,
    sales_info: pd.DataFrame,
):

    # Filter unused columns
    sales_info: pd.DataFrame = sales_info.drop("item_weighted_sales", axis=1)

    # Rename columns for better readability
    col_rename_mapper = {
        "item_cnt_day": "item_sales",
        "avg_item_price_weighted": "item_avg_price",
    }
    sales_info = sales_info.rename(columns=col_rename_mapper)

    # Replace +INF, -INF and NaN with 0
    sales_info.replace([np.inf, -np.inf], 0, inplace=True)

    print(sales_info["item_sales"].min(), sales_info["item_sales"].max())
    print(sales_info["item_avg_price"].min(), sales_info["item_avg_price"].max())

    # Do Normalization - Do range mapping
    sales_info["item_sales"] = sales_info["item_sales"].apply(np.log10)
    sales_info["item_avg_price"] = sales_info["item_avg_price"].apply(lambda x: x ** 0.125)

    print(sales_info["item_sales"].min(), sales_info["item_sales"].max())
    print(sales_info["item_avg_price"].min(), sales_info["item_avg_price"].max())
    assert False, "Test"

    # Copy inference data
    data_inference_ts = data_inference

    # Join historical sales information
    for i in range(6):
        sales_px = sales_info.query(f"date_block_num == {33 - i}")

        data_inference_ts = data_inference_ts.merge(
            sales_px,
            how="left",
            left_on=['date_block_num', 'shop_id', 'item_id', 'item_category_id'],
            right_on = ['date_block_num', 'shop_id', 'item_id', 'item_category_id'],
        )
        data_inference_ts = data_inference_ts.rename(
            columns = {
                "item_sales": f"item_sales_p{i + 1}",
                "item_avg_price": f"item_avg_price_p{i + 1}"
            },
        )

    # Replace NaN, -inf and inf with 0
    data_inference_ts = data_inference_ts.fillna(0)

    # Reorder data to fit model input
    column_inf_ts = [
        # 'date_block_num',
        'month_id',             # Month ID of P0
        'shop_id',
        'item_id',
        'item_category_id',
        # 'item_sales'
        # 'item_avg_price',
        'item_sales_p1', 'item_avg_price_p1',
        'item_sales_p2', 'item_avg_price_p2',
        'item_sales_p3', 'item_avg_price_p3',
        'item_sales_p4', 'item_avg_price_p4',
        'item_sales_p5', 'item_avg_price_p5',
        'item_sales_p6', 'item_avg_price_p6'
    ]
    data_inference_ts = data_inference_ts[column_inf_ts]

    return data_inference_ts

def split_test_set():
    pass