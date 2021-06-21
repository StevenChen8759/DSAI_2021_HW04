import numpy as np
import pandas as pd
from loguru import logger
from sklearn import preprocessing

def train_norm_old(
    input_data: pd.DataFrame,
):
    # <<<<<<<<<<<<<<<<<<<Input Data Schema>>>>>>>>>>>>>>>>>>>
    # date_block_num, shop_id, item_id, item_category_id
    # avg_sales_price, total_sales, total_record_count
    # <<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Copy From Input Data
    output_data: pd.DataFrame = input_data.copy()

    # Do normalization on average sales price
    avg_price_min = output_data["avg_sales_price"].min()
    avg_price_max = output_data["avg_sales_price"].max()
    avg_price_min_max_norm_func = (
        lambda x: (x - avg_price_min) / (avg_price_max - avg_price_min)
    )
    output_data["avg_sales_price_norm"] = (
        output_data["avg_sales_price"].apply(avg_price_min_max_norm_func)
    )
    logger.debug(
        f"Avg sales mapping: "
      + f"[{avg_price_min}, {avg_price_max}] ->"
      + f"[{output_data['avg_sales_price_norm'].min()}, {output_data['avg_sales_price_norm'].max()}]"
    )

    # Do normalization on total sales
    total_sales_min = output_data["total_sales"].min()
    total_sales_max = output_data["total_sales"].max()
    total_sales_min_max_norm_func = (
        lambda x: (x - avg_price_min) / (avg_price_max - avg_price_min)
    )
    output_data["total_sales_norm"] = (
        output_data["total_sales_norm"].apply(total_sales_min_max_norm_func)
    )
    logger.debug(
        f"Avg sales mapping: "
      + f"[{total_sales_min}, {total_sales_max}] ->"
      + f"[{output_data['total_sales_norm'].min()}, {output_data['total_sales_norm'].max()}]"
    )


def train_norm(
    input_data: pd.DataFrame,
):
    # Convert to numpy array
    input_data_np = input_data.values

    # Do Min-Max Scalar Normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    output_data_np = min_max_scaler.fit_transform(input_data_np)

    # Convert back to pandas dataframe
    output_data = pd.DataFrame(
        output_data_np,
        columns=input_data.columns,
    )
    # print(output_data)

    return output_data

def inference_norm():
    pass

def inference_denorm():
    pass

def month_norm(input_df):
    input_df["date_block_num"] = input_df["date_block_num"].apply(
        lambda x: (x % 12) - 0 / 11
    )
