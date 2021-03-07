import time
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import logging
from tensorflow.python.keras.models import load_model

from src.common.classiffication_util import print_score, print_class_report
from src.common.model_util import load_classification_model
from src.step04_load_data import load_data_into_frame


def predict(model, X_test, y_test):
    logging.info('------ Encoding data with autoencoder... ----')

    start_time = time.time()
    predictions = model.predict(X_test)
    end_time = time.time()

    print("--->Prediction time in seconds: " + str(end_time - start_time))

    print_score(y_test, predictions)
    print_class_report(y_test, predictions)
    # plot_confusion_matrix(model, X_test, y_test)
    # plt.show()

    exec_time_seconds = (time.time() - start_time)
    exec_time_minutes = exec_time_seconds / 60

    print("---> END in %s seconds = %s minutes ---" % (exec_time_seconds, exec_time_minutes))


model_dir = '../models/'
model_name = 'RandomForestClassifier'

# Load test data
data_file_name = '_benign_okiru_horiz_port_scan_ddos_1_000_000.csv_test.csv'
X_test, y_test = load_data_into_frame(data_file_name)

# Encode test data
encoder_path = model_dir + "_Encoder.h5"
encoder = load_model(encoder_path)
X_test_encode = encoder.predict(X_test)

# Load std model
print('Loading std....')
std_model = load_classification_model(model_dir + model_name + '_std.pkl')

# Predict with std model
print('Predicting with std....')
predict(std_model, X_test, y_test)

# Load ae model
print('Loading ae....')
model_ae = load_classification_model(model_dir + model_name + '_ae.pkl')

# Predict with ae model
print('Predicting with ae....')
predict(model_ae, X_test_encode, y_test)

print("The End.")
