import pandas as pd
import logging
import time

from sklearn.model_selection import train_test_split

from config import iot23_output_directory
from src.common.data_frame_util import df_get, df_drop_cols, df_clean_data, write_to_csv

# Setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Setup dataframe display options
pd.set_option('display.expand_frame_repr', False)


def clean_data(source_dir,
               file_name,
               cols_to_delete=[],
               drop_cols_with_unique_values_le_than=1,
               cols_to_category=[], export_file=False):
    start_time = time.time()

    logging.info('Loading df from ' + file_name)

    df = df_get(source_dir + file_name)
    data_frame = df_clean_data(df,
                               cols_to_delete=cols_to_delete,
                               drop_cols_with_unique_values_le_than=drop_cols_with_unique_values_le_than,
                               cols_to_category=cols_to_category, export_dir=source_dir, export_file=export_file,
                               export_file_name=file_name + "_clean.csv")

    end_time = time.time()
    exec_time_seconds = (end_time - start_time)
    exec_time_minutes = exec_time_seconds / 60
    print("---> END in %s seconds = %s minutes ---" % (exec_time_seconds, exec_time_minutes))

    return data_frame


def split_data(df, export_dir, export_file_name, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size)

    file_path_train = export_dir + export_file_name + '_train.csv'
    write_to_csv(train, file_path_train, mode='w')

    file_path_test = export_dir + export_file_name + '_test.csv'
    write_to_csv(test, file_path_test, mode='w')


cols_to_drop = ['ts', 'uid', 'id.orig_h', 'id.resp_h', 'label']
cols_to_cat = ['detailed-label']
data_file_name = '_benign_okiru_horiz_port_scan_ddos_1_000_000.csv'
data_frame = clean_data(iot23_output_directory, data_file_name, cols_to_delete=cols_to_drop, cols_to_category=cols_to_cat, export_file=True)
split_data(data_frame, iot23_output_directory, data_file_name, test_size=0.2)
