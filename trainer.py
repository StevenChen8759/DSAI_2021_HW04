from loguru import logger
import pandas as pd

from utils import csvIO, modelIO
from preprocessor import sales_feature, training_data_operation
from predictor import DTR

def main():

    total_month_count = 34
    item_count = 22170
    shop_count = 60

    logger.info("Reading dataset, file name: ./dataset/train_monthly_sales.csv")
    monthly_sales_train = csvIO.read_csv_to_pddf("./dataset/train_monthly_sales.csv")
    # print(monthly_sales_train)
    logger.debug(f"Expected: {total_month_count * item_count * shop_count}, Real: {len(monthly_sales_train)} ({len(monthly_sales_train) * 100 / (total_month_count * item_count * shop_count):.2f}%)")

    # Do feature extraction
    logger.info("Do feature extraction")
    feature_collection = training_data_operation.feature_extract(monthly_sales_train)

    # Encode Data - Input with Item ID, Shop ID and Month ID
    logger.info("[Data Composing] Build Data for model fitting")
    data_in, data_out = training_data_operation.compose_data(monthly_sales_train, feature_collection)

    decision_tree_model = DTR.train(data_in, data_out)

    modelIO.save(decision_tree_model, "dtr_8")

if __name__ == "__main__":
    main()
