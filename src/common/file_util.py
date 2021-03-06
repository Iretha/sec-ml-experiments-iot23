import glob
import logging
import os.path
from os import path


def find_files_recursively(location_dir, file_name_pattern):
    pathname = location_dir + file_name_pattern
    files = glob.glob(pathname, recursive=True)
    logging.info('Files found: ' + str(len(files)))
    return files


def filter_out_files_larger_than(files=[], max_size_in_mb=-1):
    if max_size_in_mb == -1:
        return files

    filtered_files = []
    for idx in range(len(files)):
        file = files[idx]
        file_size_in_mb = get_file_size_in_mb(file)
        if 0 < max_size_in_mb < file_size_in_mb:
            logging.info('File is too large, it will be skipped: ' + file)
        else:
            filtered_files.append(file)
    return filtered_files


def get_file_size_in_mb(file_path):
    file_exists = path.exists(file_path)
    size_in_mb = -1
    if file_exists:
        size_in_bytes = os.path.getsize(file_path)
        size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb


def write_df_to_csv(df, dest_file_path, mode='a'):
    rows = len(df.index)
    add_header = False if path.exists(dest_file_path) else True
    df.to_csv(dest_file_path, mode=mode, header=add_header, index=False)


def clean_dir_content(iot23_output_directory):
    files = glob.glob(iot23_output_directory + '/*')
    for f in files:
        os.remove(f)

    logging.info('Content of ' + iot23_output_directory + ' is deleted.')
