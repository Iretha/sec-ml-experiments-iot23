from src.common.autoenc_feature_extraction_util import create_autoenc_feature_selection_model
from src.step04_load_data import load_data_into_frame

# Load data
data_file_name = '_benign_okiru_horiz_port_scan_ddos_1_000_000.csv_train.csv'
X, y = load_data_into_frame(data_file_name)

# Use autoencoders for feature extraction
model_dir = '../models/'
model_name = '_Encoder.h5'
create_autoenc_feature_selection_model(X, y, model_dir=model_dir, model_name="_Encoder.h5")
