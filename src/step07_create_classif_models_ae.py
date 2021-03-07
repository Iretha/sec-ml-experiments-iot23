import pandas as pd
import logging
import time

from sklearn.model_selection import train_test_split
from config import iot23_output_directory
from src.common.autoenc_feature_extraction_util import create_autoenc_feature_selection_model
from src.common.classiffication_util import create_classification_models, create_classification_models_with_ae_features
from src.common.data_frame_util import df_get, df_add_cat_columns, df_corr

import warnings

# Load data
from src.step04_load_data import load_data_into_frame

data_file_name = '_benign_okiru_horiz_port_scan_ddos_1_000_000.csv_train.csv'
X, y = load_data_into_frame(data_file_name)

# Classification (Autoencoded Features)
model_dir = '../models/'
model_name = '_Encoder.h5'
encoder_path = model_dir + model_name
create_classification_models_with_ae_features(X, y, encoder_path, model_output_dir=model_dir, test_size=0.2)

# 4.3. Clustering
# logging.info('Start clustering... ')
# run_clustering(X, y, n_clusters=3)  # 0.25961017066125075

# 4.4 DL - Autoenc
# run_autoenc(X, y, "model_okiru_hori_port_scn_01.h5")  # 25 min ~ accuracy: 0.9138
