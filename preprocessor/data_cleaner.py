import pandas as pd
from loguru import logger

def remove_outlier(
    input_data: pd.DataFrame,
):
    # <<<<<<<<<<<<<<<<<<<<<<Schema of input data>>>>>>>>>>>>>>>>>>>>>>>
    # 'date_block_num', 'shop_id', 'item_id', 'item_category_id',
    # 'avg_sales_price',
    # 'total_sales',
    # 'total_record_count'
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Step 1. Do Statistic Report on Output
    logger.debug(f"Statistical Report of Input Data, Original Data Length: {len(input_data)}")
    print(f"Avg. Sales Price -> min: {input_data['avg_sales_price'].min():.2f}"
        + f", max: {input_data['avg_sales_price'].max():.2f}"
        + f", avg: {input_data['avg_sales_price'].mean():.2f}"
        + f", median: {input_data['avg_sales_price'].median():.2f}"
        + f", sd: {input_data['avg_sales_price'].std():.2f}"
    )

    print(f"Total Sales -> min: {input_data['total_sales'].min():.2f}"
        + f", max: {input_data['total_sales'].max():.2f}"
        + f", avg: {input_data['total_sales'].mean():.2f}"
        + f", median: {input_data['total_sales'].median():.2f}"
        + f", sd: {input_data['total_sales'].std():.2f}"
    )
    # print(len(input_data.query("total_sales >= 50.0")))
    # print(len(input_data.query("avg_sales_price >= 5000")))

    # Step 2. Plot boxplot with filtering outlier
    clean_data = input_data.query("total_sales <= 50.0").copy()
    logger.debug(f"Filter Total Sales Outlier, Final Length: {len(clean_data)}")

    clean_data.query("avg_sales_price <= 5000", inplace=True)
    logger.debug(f"Filter Average Sales Price Outlier, Final Length: {len(clean_data)}")

    # Step 3. Output Statistic Information
    logger.debug(f"Statistical Report of Data After Filtering Outlier")
    print(f"Avg. Sales Price -> min: {clean_data['avg_sales_price'].min():.2f}"
        + f", max: {clean_data['avg_sales_price'].max():.2f}"
        + f", avg: {clean_data['avg_sales_price'].mean():.2f}"
        + f", median: {clean_data['avg_sales_price'].median():.2f}"
        + f", sd: {clean_data['avg_sales_price'].std():.2f}"
    )

    print(f"Total Sales -> min: {clean_data['total_sales'].min():.2f}"
        + f", max: {clean_data['total_sales'].max():.2f}"
        + f", avg: {clean_data['total_sales'].mean():.2f}"
        + f", median: {clean_data['total_sales'].median():.2f}"
        + f", sd: {clean_data['total_sales'].std():.2f}"
    )

    # Step 4. Output Box Plot for Final result of filtering outlier
    boxplot_price = clean_data.boxplot(
        column=["avg_sales_price"],
    )
    boxplot_price.figure.savefig('./output/outlier_formal_boxplot_price.jpg')
    boxplot_price.clear()

    boxplot_sales = clean_data.boxplot(
        column=["total_sales"],
    )
    boxplot_sales.figure.savefig('./output/outlier_formal_boxplot_sales.jpg')
    boxplot_sales.clear()

    # Step 5. Output Histogram to view up the distribution of data
    hist_price = clean_data["avg_sales_price"].plot.hist()
    hist_price.figure.savefig('./output/outlier_formal_hist_price.jpg')
    hist_price.clear()

    hist_sales = clean_data["total_sales"].plot.hist()
    hist_sales.figure.savefig('./output/outlier_formal_hist_sales.jpg')
    hist_sales.clear()

    # Step Final. return data
    # print(len(clean_data.query("avg_sales_price >= 0.0 and avg_sales_price <= 0.5")))
    return clean_data

    # logger.debug("Box Plot on Training Data")
    # boxplot_sales = clean_data.boxplot(
    #     column=["total_sales"],
    # )
    # boxplot_sales.figure.savefig('./output/outlier_boxplot_sales_less_10.jpg')
    # hist_sales = clean_data["total_sales"].plot.hist()
    # hist_sales.figure.savefig('./output/outlier_rm_hist_sales_less_100.jpg')

    # boxplot_avg_price = clean_data.boxplot(
    #     column=["avg_sales_price"],
    # )
    # boxplot_avg_price.figure.savefig('./output/outlier_boxplot_price_less_1500.jpg')