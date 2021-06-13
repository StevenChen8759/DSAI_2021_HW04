from datetime import datetime
from typing import List
import sys

import pandas as pd
import numpy as np
from loguru import logger

from utils import csvIO, plotter


def query_original_record_count(
    data: pd.DataFrame,
    category_info: pd.DataFrame,
    groupby_field: List[str],
    opfilename: str
):
    result = data.groupby(groupby_field).agg("count").rename({"date": "count"}, axis=1)
    result = result[["count"]]

    zero_index_list = [[idx, 0] for idx in range(22170) if idx not in result.index]
    zero_raw_to_add = pd.DataFrame(zero_index_list, columns = ["item_id", "count"])
    result = result.reset_index()
    result = pd.concat([result, zero_raw_to_add]).reset_index(drop=True).sort_values("item_id")

    # Plot data count
    result_np = result["count"].to_numpy()
    plotter.plot_data_count(result_np, f"{opfilename}_plot.jpg")

    # Aggregate items with filter which satisfy specififc count of data

    # Join with category of items

    # Output data count to csv file
    csvIO.write_pd_to_csv(result, f"{opfilename}_raw.csv", False)


def monthly_sales_aggregation(
    train_data: pd.DataFrame,
    item_category: pd.DataFrame,
):
    # Train data - month adjustment
    train_data["month_no"] = train_data["date_block_num"].apply(lambda x: (x % 12) + 1)

    # Train data - calculate total sales without refunded items
    train_data["item_weighted_sales"] = train_data["item_price"] * train_data["item_cnt_day"]

    # Train data - do statistics
    aggregate_map = {"item_cnt_day": "sum", "item_weighted_sales": "sum"}

    logger.info("Start Aggregation...")
    train_monthly_sales: pd.DataFrame = train_data.groupby(["date_block_num", "shop_id", "item_id"]).agg(aggregate_map).reset_index()
    logger.info("Done...")
    train_monthly_sales["avg_item_price_weighted"] = train_monthly_sales["item_weighted_sales"] / train_monthly_sales["item_cnt_day"]

    # Train data - add category information by join operation
    train_monthly_sales = train_monthly_sales.join(item_category, on="item_id")

    # Train data - reorder columns
    sales_order = ["date_block_num",
                   "shop_id",
                   "item_id",
                   "item_category_id",
                   "item_cnt_day",
                   "avg_item_price_weighted",
                   "item_weighted_sales",
                   ]
    train_monthly_sales = train_monthly_sales[sales_order]
    print(train_monthly_sales)

    csvIO.write_pd_to_csv(train_monthly_sales, "train_monthly_sales.csv", False)


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

    # print(train_data)
    print(category_of_item[32])

    # query_original_record_count(train_data, category_of_item, ["item_id"], "item_id_count")

    monthly_sales_aggregation(train_data, category_of_item)
