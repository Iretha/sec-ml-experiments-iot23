import time
import logging


from src.common.data_frame_util import join_data_slices_in_single_data_frame, write_to_csv


# Setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def join_data_slices_from_files(source_dir, source_files, output_file_name, replace_values={}, slice_size=1000):
    logging.info('Start joining slices... ')

    start_time = time.time()

    # make single dataframe from multiple files
    data_frame = join_data_slices_in_single_data_frame(source_dir, source_files, slice_size=slice_size)

    # Replacing empty values with the values from the dictionary
    if len(replace_values) > 0:
        data_frame.applymap(lambda s: replace_values.get(s) if s in replace_values else s)

    # write to file
    write_to_csv(data_frame, source_dir + output_file_name, mode='w')

    logging.info('Output file is: ' + source_dir + output_file_name)

    end_time = time.time()
    exec_time_seconds = (end_time - start_time)
    exec_time_minutes = exec_time_seconds / 60
    print("---> END in %s seconds = %s minutes ---" % (exec_time_seconds, exec_time_minutes))


# dict_empty_values = {
#     '(empty)': -999,
#     '-': -999,
#     np.nan: -999
# }
# source_traffic_files = [
#     'Benign.csv',
#     'DDoS.csv'
# ]
# target_file_name = '_benign_ddos_2000.csv'
# join_data_slices_from_files(iot23_output_directory, source_traffic_files, target_file_name, replace_values=dict_empty_values, slice_size=1000)
