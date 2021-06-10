from datetime import datetime
from typing import List
import sys

import pandas as pd
import numpy as np
from loguru import logger

from utils import csvIO, plotter


def query_record_count(data: pd.DataFrame,
                       groupby_field: List[str],
                       opfilename: str
                       ):
    result = data.groupby(groupby_field).agg("count").rename({"date": "count"}, axis=1)

    for i in range(22170):
        if i not in result.index:
            result["count"][i] = 0
    result = result.sort_index()
    result = result[["count"]]

    # Plot data count
    result_np = result.to_numpy()
    print(np.max(result_np))
    plotter.plot_data_count(result_np, f"{opfilename}_plot.jpg")

    # Output data count to csv file
    csvIO.write_pd_to_csv(result, f"{opfilename}_raw.csv")


def wavg(input_series, origin_df, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = input_series
    w = origin_df[weight_name].loc[input_series]

    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


if __name__ == "__main__":

    item_idx = csvIO.read_csv_to_pddf("./dataset/items.csv")

    # Query category number (0 ~ 83) of an item by index
    category_of_item = item_idx["item_category_id"]

    month_base = datetime(2013, 1, 1)
    train_data = csvIO.read_csv_to_pddf("./dataset/sales_train.csv")

    # Do SQL query pn DataFrame to reorder data
    train_data.sort_values(["shop_id", "date_block_num"],
                            inplace=True,
                            ignore_index=True
                            )

    print(train_data)

    # query_record_count(train_data, ["item_id"], "item_id_count")

    # Train data - month adjustment
    train_data["month_no"] = train_data["date_block_num"].apply(lambda x: (x % 12) + 1)

    # Train data - calculate total sales without refunded items
    train_data["item_weighted_sales"] = train_data["item_price"] * train_data["item_cnt_day"]

    # Train data - do statistics
    aggregate_map = {"item_cnt_day": "sum", "item_weighted_sales": "sum"}

    logger.info("Start Aggregation...")
    train_monthly_sales = train_data.groupby(["date_block_num", "shop_id", "item_id"]).agg(aggregate_map).reset_index()
    logger.info("Done...")
    print(train_monthly_sales)
    csvIO.write_pd_to_csv(train_monthly_sales, "train_monthly_sales.csv")
