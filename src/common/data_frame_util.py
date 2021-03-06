import logging
import pandas as pd


def df_load_dataframe_from_bro_file(file_path, skip_rows=7):
    logging.info('\tLoad df from bro file: ' + file_path)
    header_names = pd.read_csv(file_path, sep="\s+", skiprows=skip_rows - 1, nrows=1, usecols=range(1, 24))
    read_file = pd.read_csv(file_path, sep="\s+", skiprows=skip_rows, comment="#")
    read_file.columns = header_names.columns
    return read_file






