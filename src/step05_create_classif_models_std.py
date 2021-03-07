from src.common.classiffication_util import create_classification_models
from src.step04_load_data import load_data_into_frame

# Load data
data_file_name = '_benign_okiru_horiz_port_scan_ddos_1_000_000.csv_train.csv'
X, y = load_data_into_frame(data_file_name)

# Run Classification (Standard)
model_dir = '../models/'
create_classification_models(X, y, model_output_dir=model_dir)
