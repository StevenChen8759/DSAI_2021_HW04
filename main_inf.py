from loguru import logger
import pandas as pd

from utils import csvIO, modelIO
from preprocessor import sales_feature, training_data_operation
from predictor import DTR

if __name__ == "__main__":

    # load testing csv file
    inference_data = csvIO.read_csv_to_pddf("./dataset/test.csv")
    print(inference_data)

    # load model
    dtf_8 = modelIO.load("./output/dtr_8.model")

    # do inference with input data
    result = DTR.inference(dtf_8, inference_data)

    opres = pd.DataFrame(result, columns=["item_cnt_day"]).reset_index().rename(columns={"index": "ID", "item_cnt_day": "item_cnt_month"})
    print(opres)
    csvIO.write_pd_to_csv(opres, "submission.csv", False)
