import pandas as pd
import logging
import time

from sklearn.model_selection import train_test_split
from config import iot23_output_directory
from src.common.autoenc_feature_extraction_util import create_autoenc_feature_selection_model
from src.common.classiffication_util import create_classification_models
from src.common.data_frame_util import df_get, df_add_cat_columns, df_corr

import warnings

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

# Setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Setup dataframe display options
pd.set_option('display.expand_frame_repr', False)


def load_data_into_frame(data_file_name):
    # 0. Load dataframe
    logging.info('Loading clean df from ' + data_file_name)
    df = df_get(iot23_output_directory + data_file_name)
    # print(df.dtypes)

    # 1. Convert objects to categorical data
    object_column_names = list(df.select_dtypes(include=['object']).columns)
    df_add_cat_columns(df, object_column_names)
    cat_map = dict(enumerate(df['detailed-label'].cat.categories))
    print(cat_map)

    # 3. Select features
    all_non_object_cols = list(df.select_dtypes(exclude=['object', 'category']).columns)
    selected_features = [x for x in all_non_object_cols if x not in ['detailed-label', 'detailed-label_cat']]
    y = df['detailed-label_cat']
    X = df[selected_features]
    print(selected_features)
    return X, y

# data_file_name = '_benign_okiru_horiz_port_scan_ddos_1_000_000.csv_train.csv'
# load_data_into_frame(data_file_name)
