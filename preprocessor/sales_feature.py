# Provide a statistical feature of sales with different features
from typing import List

import numpy as np
import pandas as pd
from loguru import logger


def evaluate_mean(
    train_data: pd.DataFrame,
    input_param: List[str],
    output_feature: List[str],
):
    """Do Statistic on average salse based on different item at each month

    Args:
        train_data (pd.DataFrame): training data for feature extraction
    """
    # Evaluate month ID
    convert_month_id = lambda x: (x % 12) + 1
    train_data["month_id"] = train_data["date_block_num"].apply(convert_month_id)

    # Replace date_block_num with month_id if this parameter is in columns list
    for i in range(len(input_param)):
        if input_param[i] == "date_block_num":
            input_param[i] = "month_id"

    # Select only necessary columns
    columns: List[str] = input_param + output_feature
    train_data_agg: pd.DataFrame = train_data[columns]

    # Do aggregation with item ID and date_block_num
    stat_res: pd.DataFrame = train_data_agg.groupby(input_param).mean().reset_index()

    # return final result
    return stat_res


def shop_seasonal_sales_of_category(
    input_data: pd.DataFrame
):
    # <<<<<<<<<<<<<<<<<<<<<<Schema of input data>>>>>>>>>>>>>>>>>>>>>>>
    # 'date_block_num', 'shop_id', 'item_id', 'item_category_id,
    # 'avg_sales_price',
    # 'total_sales',
    # 'total_record_count'
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Step 1. Do aggregation of input data
    aggregate_map = {
        "total_sales": ["sum", "mean"],
    }

    stat_table: pd.DataFrame = (
        input_data.groupby(['date_block_num', 'shop_id', 'item_category_id'])
                  .agg(aggregate_map)
    )

    # Step 2. Integrate output columns
    stat_table.columns = stat_table.columns.map('_'.join).str.strip('_')

    # Step 3. Zero Patch
    logger.debug("Do zero patching...")

    # Collect index to patched
    index_to_patch = []
    for i in range(34):
        for j in range(60):
            for k in range(84):
                iter_index = (i, j, k)
                if not iter_index in stat_table.index:
                    index_to_patch.append(iter_index)
    value_to_patch = [[0.0, 0.0] for _ in index_to_patch]

    # Create multi-index object
    patch_df_index = pd.MultiIndex.from_tuples(
        index_to_patch,
        names=[
            "date_block_num",
            "shop_id",
            "item_category_id",
        ]
    )

    # Create Dataframe to patched
    patch_df = pd.DataFrame(
        data=value_to_patch,
        index=patch_df_index,
        columns=["total_sales_sum", "total_sales_mean"],
    )

    # Merge DataFrame to finish patching
    stat_table = pd.concat([stat_table, patch_df]).rename(
        columns={
            "total_sales_sum": "cat_shop_total_sales_sum",
            "total_sales_mean": "cat_shop_total_sales_mean",
        }
    )
    logger.debug("Done")

    # Sort data by index, then reset index
    stat_table.sort_index(inplace=True)
    stat_table.reset_index(inplace=True)

    return stat_table


def item_total_sales(input_data: pd.DataFrame):
    # <<<<<<<<<<<<<<<<<<<<<<Schema of input data>>>>>>>>>>>>>>>>>>>>>>>
    # 'date_block_num', 'shop_id', 'item_id',
    # 'avg_sales_price',
    # 'total_sales',
    # 'total_record_count'
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Step 1. Do aggregation of input data
    aggregate_map = {
        "total_sales": ["sum", "mean"],
    }

    stat_table: pd.DataFrame = (
        input_data.groupby(['date_block_num', 'item_id',])
                  .agg(aggregate_map)
    )

    # Step 2. Integrate output columns
    stat_table.columns = stat_table.columns.map('_'.join).str.strip('_')


    # Step 3. Zero Patch
    logger.debug("Do zero patching...")

    # Collect index to patched
    index_to_patch = []
    for i in range(34):
        for j in range(22170):
                iter_index = (i, j)
                if not iter_index in stat_table.index:
                    index_to_patch.append(iter_index)
    value_to_patch = [[0.0, 0.0] for _ in index_to_patch]

    # Create multi-index object
    patch_df_index = pd.MultiIndex.from_tuples(
        index_to_patch,
        names=[
            "date_block_num",
            "item_id",
        ]
    )

    # Create Dataframe to patched
    patch_df = pd.DataFrame(
        data=value_to_patch,
        index=patch_df_index,
        columns=["total_sales_sum", "total_sales_mean"],
    )

    # Merge DataFrame to finish patching
    stat_table = pd.concat([stat_table, patch_df]).rename(
        columns={
            "total_sales_sum": "monthly_total_item_sales_sum",
            "total_sales_mean": "monthly_total_item_sales_mean",
        }
    )
    logger.debug("Done")

    # Sort data by index, then reset index
    stat_table.sort_index(inplace=True)
    stat_table.reset_index(inplace=True)

    return stat_table


def item_category_total_sales(
    input_data: pd.DataFrame
):
    # <<<<<<<<<<<<<<<<<<<<<<Schema of input data>>>>>>>>>>>>>>>>>>>>>>>
    # 'date_block_num', 'shop_id', 'item_id', 'item_category_id,
    # 'avg_sales_price',
    # 'total_sales',
    # 'total_record_count'
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Step 1. Do aggregation of input data
    aggregate_map = {
        "total_sales": ["sum", "mean"],
    }

    stat_table: pd.DataFrame = (
        input_data.groupby(['date_block_num', 'item_category_id',])
                  .agg(aggregate_map)
    )

    # Step 2. Integrate output columns
    stat_table.columns = stat_table.columns.map('_'.join).str.strip('_')


    # Step 3. Zero Patch
    logger.debug("Do zero patching...")

    # Collect index to patched
    index_to_patch = []
    for i in range(34):
        for j in range(84):
                iter_index = (i, j)
                if not iter_index in stat_table.index:
                    index_to_patch.append(iter_index)
    value_to_patch = [[0.0, 0.0] for _ in index_to_patch]

    # Create multi-index object
    patch_df_index = pd.MultiIndex.from_tuples(
        index_to_patch,
        names=[
            "date_block_num",
            "item_category_id",
        ]
    )

    # Create Dataframe to patched
    patch_df = pd.DataFrame(
        data=value_to_patch,
        index=patch_df_index,
        columns=["total_sales_sum", "total_sales_mean"],
    )

    # Merge DataFrame to finish patching
    stat_table = pd.concat([stat_table, patch_df]).rename(
        columns={
            "total_sales_sum": "monthly_total_item_cat_sales_sum",
            "total_sales_mean": "monthly_total_item_cat_sales_mean",
        }
    )
    logger.debug("Done")

    # Sort data by index, then reset index
    stat_table.sort_index(inplace=True)
    stat_table.reset_index(inplace=True)

    return stat_table

def category_shop(train_data: pd.DataFrame):
    pass


if __name__ == "__main__":
    pass