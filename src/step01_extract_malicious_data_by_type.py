import logging
import time

from config import iot23_dataset_location, iot23_output_directory, iot23_file_name_pattern, iot23_malicious_type_column_name
from src.common.data_frame_util import df_load_dataframe_from_bro_file
from src.common.file_util import find_files_recursively, filter_out_files_larger_than, write_df_to_csv, clean_dir_content

# Setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def split_files_by_malicious_type(dataset_location, file_name_pattern, class_col_name, output_dir, skip_files_larger_than=-1):
    start_time = time.time()

    dataset_files = find_files_recursively(dataset_location, file_name_pattern)
    dataset_files = filter_out_files_larger_than(dataset_files, max_size_in_mb=skip_files_larger_than)

    for file_path in dataset_files:
        logging.info('Start processing file: ' + file_path)
        data_frame = df_load_dataframe_from_bro_file(file_path, skip_rows=7)  # load dataframe, skip first rows
        data_frame[class_col_name].replace(to_replace='-', value='Benign', inplace=True)  # Label benign traffic
        unique_values = data_frame[class_col_name].unique()  # get classification unique values
        for value in unique_values:
            logging.info('\tProcessing ' + value + " from  " + file_path)
            df_by_value = data_frame.loc[data_frame[class_col_name] == value]
            dest_file_path = output_dir + value + ".csv"
            write_df_to_csv(df_by_value, dest_file_path)
        logging.info('End processing file: ' + file_path)

    end_time = time.time()
    exec_time_seconds = (end_time - start_time)
    exec_time_minutes = exec_time_seconds / 60
    print("---> END in %s seconds = %s minutes ---" % (exec_time_seconds, exec_time_minutes))


clean_dir_content(iot23_output_directory)
split_files_by_malicious_type(iot23_dataset_location, iot23_file_name_pattern, iot23_malicious_type_column_name, iot23_output_directory)
