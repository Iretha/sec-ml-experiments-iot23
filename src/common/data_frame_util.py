import logging
import pandas as pd
import os.path
from os import path

from sklearn.preprocessing import OrdinalEncoder


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
    add_header = False if (mode == 'a' and path.exists(dest_file_path)) else True
    df.to_csv(dest_file_path, mode=mode, header=add_header, index=False)


def df_drop_cols(df, *cols):
    for col in cols:
        df.drop(columns=col, inplace=True)
    # for colIdx in cols:
    #     df = df.drop(df.iloc[:, colIdx:colIdx+1], inplace=True, axis=1)


def df_clean_data(df, cols_to_delete=[], drop_cols_with_unique_values_le_than=1, cols_to_category=[], export_file=False, export_dir=None, export_file_name=None):
    logging.info('Start cleaning...')

    # Print stats before
    logging.info('Print stats before cleaning: ')
    print(df.shape)
    print(df.dtypes)
    print(df.nunique())

    # 1. Delete columns by name
    logging.info('Deleting cols: ' + ', '.join(cols_to_delete))
    df_drop_cols(df, cols_to_delete)

    # 2. Drop duplicated rows
    logging.info('Removing duplicated rows ')
    df = df.drop_duplicates()

    # 3. Delete columns with not enough unique values
    delete_columns_with_unique_values_less_than(df, le_than=drop_cols_with_unique_values_le_than)

    # 4. Convert to categorical
    df_add_cat_columns(df, column_names=cols_to_category)
    print(df.dtypes)

    # # 5. Convert to numeric all columns that can be converted
    # df = df_transform_obj_types_to_numeric(df)

    # 6. Encode all object types, that cannot be converted to numeric
    df_encode_obj_types_with_ordinal_encoder(df)

    # 7. Select non-object columns
    columns_names = list(df.select_dtypes(exclude=['object']).columns)
    df = df[columns_names]

    # 8. Save to file
    if export_file:
        write_to_csv(df, export_dir + export_file_name, mode='w')

    # Print stats after
    logging.info('Print stats after cleaning: ')
    print(df.shape)
    print(df.dtypes)
    print(df.nunique())

    return df


# Remove columns with a single value
def delete_columns_with_unique_values_less_than(df, le_than=1):
    column_names = list(df.columns)
    columns_to_drop = []
    for column_name in column_names:
        unique_values = len(df[column_name].value_counts(dropna=True))
        if unique_values <= le_than:
            columns_to_drop.append(column_name)

    df_drop_cols(df, columns_to_drop)
    logging.info(
        'Columns with unique values less or equals than ' + str(le_than) + ' removed: ' + ', '.join(columns_to_drop))


def df_transform_obj_types_to_numeric(df):
    object_column_names = list(df.select_dtypes(include=['object']).columns)
    for object_column_name in object_column_names:
        df[object_column_name] = pd.to_numeric(df[object_column_name], errors='ignore')
    return df


def df_encode_obj_types_with_ordinal_encoder(df):
    ord_enc = OrdinalEncoder()
    object_column_names = list(df.select_dtypes(include=['object']).columns)
    for object_column_name in object_column_names:
        df[object_column_name] = pd.to_numeric(df[object_column_name], errors='ignore')
        if df[object_column_name].dtype == 'object':
            code_col_name = str(object_column_name) + '_encoded'
            try:
                df[object_column_name].replace(-9999, '-', inplace=True)
                df[code_col_name] = ord_enc.fit_transform(df[[object_column_name]])
            except:
                logging.err('Could not encode column: ' + code_col_name)


def df_add_cat_columns(df, column_names=[]):
    for column_name in column_names:
        code_col_name = str(column_name) + '_cat'
        df[column_name] = df[column_name].astype('category')
        df[code_col_name] = df[column_name].cat.codes
