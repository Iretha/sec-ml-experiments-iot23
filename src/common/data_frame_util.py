import logging
import pandas as pd
import os.path
from os import path


def df_get(file_path):
    logging.info('\tLoading df from ' + file_path)
    return pd.read_csv(file_path)


def df_load_dataframe_from_bro_file(file_path, skip_rows=7):
    logging.info('\tLoad df from bro file: ' + file_path)
    header_names = pd.read_csv(file_path, sep="\s+", skiprows=skip_rows - 1, nrows=1, usecols=range(1, 24))
    read_file = pd.read_csv(file_path, sep="\s+", skiprows=skip_rows, comment="#")
    read_file.columns = header_names.columns
    return read_file


def join_data_slices_in_single_data_frame(source_dir, source_files, slice_size=0, head=True):
    df_list = []
    for source_file in source_files:
        df = df_get(source_dir + source_file)
        print(df.shape)
        df = df if slice_size == 0 else (df.head(slice_size) if head else df.tail(slice_size))
        df_list.append(df)
    return pd.concat(df_list)


def write_to_csv(df, dest_file_path, mode='a'):
    add_header = False if path.exists(dest_file_path) else True
    df.to_csv(dest_file_path, mode=mode, header=add_header, index=False)
