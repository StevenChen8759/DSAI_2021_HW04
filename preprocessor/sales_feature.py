# Provide a statistical feature of sales with different features
from typing import List

import numpy as np
import pandas as pd

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


def category_month(train_data: pd.DataFrame):
    pass


def item_shop(train_data: pd.DataFrame):
    pass


def category_shop(train_data: pd.DataFrame):
    pass


if __name__ == "__main__":
    pass