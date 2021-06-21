import numpy as np
import pandas as pd
from loguru import logger
from pandarallel import pandarallel

from utils import csvIO

def integrate_monthly_sales(
    input_data: pd.DataFrame,
) -> pd.DataFrame:
    # <<<<<<<<<<<<<<<<<<<<<<Schema of input data>>>>>>>>>>>>>>>>>>>>>>
    # date, date_block_num, shop_id, item_id, item_price, item_cnt_day
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Step 1. add month ID into input data
    input_data["month_no"] = input_data["date_block_num"].apply(lambda x: (x % 12) + 1)

    # Step 2. reset index for counting number of record
    input_data = input_data.reset_index()

    # Step 3. aggregate sum of general sales and its average with count of data
    logger.debug("Aggregation for general sales...")
    aggregate_map = {
        "index": "count",       # Total Record Count
        "item_cnt_day": "sum",  # Total Sales
        "item_price": "mean"    # Average Price
    }

    general_sales: pd.DataFrame = (
        input_data.loc[input_data["item_cnt_day"] > 0]
                  .groupby(["date_block_num", "shop_id", "item_id"])
                  .agg(aggregate_map)
    )

    general_sales.rename(
        columns={
            "index": "general_record_count",
            "item_cnt_day": "general_sales_count",
            "item_price": "avg_sales_price",
        },
        inplace=True,
    )

    # Step 4. aggregate sum of refund sales and its average with count of data
    aggregate_map = {
        "index": "count",       # Total Record Count
        "item_cnt_day": "sum",  # Total Sales
    }

    logger.debug("Aggregation for refund sales...")
    refund_sales: pd.DataFrame = (
        input_data.loc[input_data["item_cnt_day"] < 0]
                  .groupby(["date_block_num", "shop_id", "item_id"])
                  .agg(aggregate_map)
    )
    refund_sales.rename(
        columns={
            "item_cnt_day": "refund_sales_count",
            "index": "refund_record_count",
        },
        inplace=True,
    )

    # Step 5. merge general and refund sales, then calculate real sales and record count
    logger.debug("Evaluate total sales...")
    agg_sales = general_sales.join(
        refund_sales,
        how='left',
    ).fillna(0)
    agg_sales["total_sales"] = agg_sales["general_sales_count"] + agg_sales["refund_sales_count"]
    agg_sales["total_record_count"] = agg_sales["general_record_count"] + agg_sales["refund_record_count"]

    # Step 6. Remove non-positive sales value
    agg_sales.query("total_sales > 0", inplace=True)

    # Step 7. Remove unused columns and return data with reset index.
    agg_sales.drop(
        columns=[
            "general_sales_count",
            "general_record_count",
            "refund_sales_count",
            "refund_record_count",
        ],
        inplace=True
    )
    agg_sales.reset_index(inplace=True)

    if not agg_sales.isin([np.inf]).values.any():
        logger.success("No element with value +inf")
    else:
        logger.error("!!!!!Found element with value +inf!!!!!")

    if not agg_sales.isin([-np.inf]).values.any():
        logger.success("No element with value -inf")
    else:
        logger.error("!!!!!Found element with value -inf!!!!!")

    if not agg_sales.isin([np.nan]).values.any():
        logger.success("No element with value NaN")
    else:
        logger.error("!!!!!Found element with value NaN!!!!!")

    # <<<<<<<<<<<<<<<<<<<<<<Schema of output data>>>>>>>>>>>>>>>>>>>>>>>
    # 'date_block_num', 'shop_id', 'item_id',
    # 'avg_sales_price',
    # 'total_sales',
    # 'total_record_count'
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return agg_sales


def join_category_info(
    input_data: pd.DataFrame,
    item_data: pd.DataFrame
) -> pd.DataFrame:
    # Build-up item-category mapper
    item_category_map = item_data["item_category_id"]

    # Return data joined with category
    return input_data.join(item_category_map, on="item_id")


def feature_join(
    input_data: pd.DataFrame,
    cat_shop_sales_heat_feature: pd.DataFrame,
    monthly_item_heat_feature: pd.DataFrame,
    monthly_item_cat_heat_feature: pd.DataFrame,
    drop_column: bool = True,
) -> pd.DataFrame:
    # <<<<<<<<<<<<<<<<<<<<< Schema of input data >>>>>>>>>>>>>>>>>>>>>>
    # 'date_block_num', 'shop_id', 'item_id', 'item_category_id',
    # 'avg_sales_price',
    # 'total_sales', 'cat_shop_total_sales_mean',
    # 'total_record_count'
    # <<<<<<<<<<<<<<<<<<<<<< Heat Feature Schema >>>>>>>>>>>>>>>>>>>>>>
    # 1. Cat Shop
    # date_block_num, shop_id, item_category_id,
    # cat_shop_total_sales_sum, cat_shop_total_sales_mean,
    # cat_shop_sales_heat
    #
    # 2. Monthly Item
    # date_block_num, item_id,
    # monthly_total_item_sales_sum,
    # monthly_total_item_sales_mean,
    # monthly_total_item_sales_heat
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # pandarallel.initialize()
    # print(sales_heat_feature.columns)

    # Remove unnecessary column of sales_heat_feature
    # if drop_column:
    #     cat_shop_sales_heat_feature.drop(
    #         columns=['cat_shop_total_sales_sum',],
    #         inplace=True
    #     )
    #     monthly_item_heat_feature.drop(
    #         columns=['monthly_total_item_sales_sum',],
    #         inplace=True
    #     )

    # Join Sales Heat Feature
    output_data = (
        input_data.merge(
            cat_shop_sales_heat_feature,
            left_on=[
                'date_block_num',
                'shop_id',
                'item_category_id',
            ],
            right_on=[
                'date_block_num',
                'shop_id',
                'item_category_id',
            ],
            how='left',
        )
    )
    assert not output_data.isna().values.any(), "Output value contains NaN after merging feature"

    # Join Month Item Feature
    output_data = (
        output_data.merge(
            monthly_item_heat_feature,
            left_on=[
                'date_block_num',
                'item_id',
            ],
            right_on=[
                'date_block_num',
                'item_id',
            ],
            how='left',
        )
    )
    assert not output_data.isna().values.any(), "Output value contains NaN after merging feature"

    # Join Month Item Category Feature
    output_data = (
        output_data.merge(
            monthly_item_cat_heat_feature,
            left_on=[
                'date_block_num',
                'item_category_id',
            ],
            right_on=[
                'date_block_num',
                'item_category_id',
            ],
            how='left',
        )
    )
    assert not output_data.isna().values.any(), "Output value contains NaN after merging feature"

    # Drop duplicated data
    # output_data.drop_duplicates(
    #     subset=['index'],
    #     inplace=True,
    # )

    # output_data.fillna(0, inplace=True)

    # print(output_data.isna().values.any())
    # <<<<<<<<<<<<<<<<<<<<< Schema of output data >>>>>>>>>>>>>>>>>>>>>>
    # 'date_block_num', 'shop_id', 'item_id', 'item_category_id',
    # 'avg_sales_price',
    # 'total_sales',
    # 'total_record_count', 'cat_shop_sales_heat'
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    return output_data


def encode_time_series_data(
    input_data: pd.DataFrame,
    month_count: int = 12,
    base_month: int = 0,            # Fetch Data Started From 2013-01 (Default)
) -> pd.DataFrame:
    # <<<<<<<<<<<<<<<<<<<Input Data Schema>>>>>>>>>>>>>>>>>>>
    # date_block_num, shop_id, item_id, item_category_id
    # avg_sales_price, total_sales, total_record_count,
    # 'total_record_count',
    # 'cat_shop_sales_heat',
    # 'monthly_total_item_sales_heat',
    # <<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Filter months which is not able to generate complete time series data
    if base_month > month_count:
        output_ts_data = input_data.query(f"date_block_num >= {base_month}")
    else:
        output_ts_data = input_data.query(f"date_block_num >= {month_count}")

    # Join historical sales information - join data i-th month ago
    for i in range(1, month_count + 1):
        # if month_count > base_month:
        #     last_i_month_info = input_data.query(f"date_block_num <= {33 - i} and date_block_num >= {base_month - month_count}")
        # else:
        last_i_month_info = input_data.query(f"date_block_num <= {33 - i}")
        last_i_month_info["date_block_num"] = last_i_month_info["date_block_num"].apply(lambda x: x + i)   # shift date_block_num

        output_ts_data = (
            output_ts_data.merge(
                last_i_month_info,
                how="left",
                left_on=['date_block_num', 'shop_id', 'item_id', 'item_category_id'],
                right_on = ['date_block_num','shop_id', 'item_id', 'item_category_id'],
                suffixes=["", f"_p{i}"],
                copy=False,
            )
        )

        if i != 1 and i != 12 and i != 24:
            output_ts_data.drop(
                columns=[
                    f'total_sales_p{i}',
                    f'avg_sales_price_p{i}',
                    f'cat_shop_sales_heat_p{i}',
                    f'monthly_total_item_sales_heat_p{i}',
                    f'monthly_total_item_cat_sales_heat_p{i}',
                ],
                inplace=True,
            )
        elif i == 12:
            output_ts_data.drop(
                columns=[
                    # f'total_sales_p{i}',
                    f'avg_sales_price_p{i}',
                    # f'cat_shop_sales_heat_p{i}',
                    # f'monthly_total_item_sales_heat_p{i}',
                    # f'monthly_total_item_cat_sales_heat_p{i}',
                ],
                inplace=True,
            )
        elif i == 24:
            output_ts_data.drop(
                columns=[
                    # f'total_sales_p{i}',
                    f'avg_sales_price_p{i}',
                    # f'cat_shop_sales_heat_p{i}',
                    # f'monthly_total_item_sales_heat_p{i}',
                    # f'monthly_total_item_cat_sales_heat_p{i}',
                ],
                inplace=True,
            )

    # Replace NaN with 0
    output_ts_data = output_ts_data.fillna(0)

    return output_ts_data

def make_inference_ts_data(
    infernece_data: pd.DataFrame,
    monthly_sales_info: pd.DataFrame,
    month_count: int = 12,          # Last 12 Month Data3
    inf_date_block_num: int = 34,   # 2015-11
):
    # Copy Prototype of merged dataframe
    output_ts_data = infernece_data.copy()
    output_ts_data["date_block_num"] = output_ts_data["ID"].apply(lambda x: 34)

    # Join historical sales information - join data i-th month ago
    for i in range(1, month_count + 1):
        last_i_month_info = monthly_sales_info.query(f"date_block_num <= {inf_date_block_num - i}")
        last_i_month_info["date_block_num"].apply(lambda x: x + i)   # shift date_block_num

        output_ts_data = (
            output_ts_data.merge(
                last_i_month_info,
                how="left",
                left_on=['date_block_num', 'shop_id', 'item_id', 'item_category_id'],
                right_on = ['date_block_num', 'shop_id', 'item_id', 'item_category_id'],
                suffixes=["", f"_p{i}"],
                copy=False,
            )
        )

        if i != 1 and i != 12 and i != 24:
            output_ts_data.drop(
                columns=[
                    f'total_sales_p{i}',
                    f'avg_sales_price_p{i}',
                    f'cat_shop_sales_heat_p{i}',
                    f'monthly_total_item_sales_heat_p{i}',
                    f'monthly_total_item_cat_sales_heat_p{i}',
                ],
                inplace=True,
            )
        elif i == 12:
            output_ts_data.drop(
                columns=[
                    # f'total_sales_p{i}',
                    f'avg_sales_price_p{i}',
                    # f'cat_shop_sales_heat_p{i}',
                    # f'monthly_total_item_sales_heat_p{i}',
                    # f'monthly_total_item_cat_sales_heat_p{i}',
                ],
                inplace=True,
            )
        elif i == 24:
            output_ts_data.drop(
                columns=[
                    # f'total_sales_p{i}',
                    f'avg_sales_price_p{i}',
                    # f'cat_shop_sales_heat_p{i}',
                    # f'monthly_total_item_sales_heat_p{i}',
                    # f'monthly_total_item_cat_sales_heat_p{i}',
                ],
                inplace=True,
            )

    # Replace NaN with 0
    output_ts_data = output_ts_data.fillna(0)

    return output_ts_data
