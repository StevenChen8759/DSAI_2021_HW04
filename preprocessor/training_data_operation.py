from typing import Dict

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
#        'month_id',
#        'shop_id',
#        'item_id',
#        'item_category_id',
        'item_cnt_day_item_month',
        # 'item_cnt_day_category_month',
        'item_cnt_day_item_shop',
        # 'item_cnt_day_category_shop'
        # 'avg_item_price_weighted',
        # 'item_sales_last_month',
        ]
    data_in = data_compose[columns_in]

    columns_out = [
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

def split_test_set():
    pass