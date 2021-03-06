import numpy as np


from config import iot23_output_directory
from src.step02_join_data_slices_for_experiment import join_data_slices_from_files

# 1. Create a single file from different traffic sources with different attack types
dict_empty_values = {
    '(empty)': -999,
    '-': -999,
    np.nan: -999
}
source_traffic_files = [
    'Benign.csv',
    'Okiru.csv',
    'PartOfAHorizontalPortScan.csv',
    'DDoS.csv'
]
slice_size = 1_000_000
target_file_name = '_benign_okiru_horiz_port_scan_ddos_1_000_000.csv'
join_data_slices_from_files(iot23_output_directory, source_traffic_files, target_file_name, replace_values=dict_empty_values, slice_size=slice_size)
