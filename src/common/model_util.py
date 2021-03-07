import pickle
import logging


# Save model
def save_classification_model(model, model_dir=None, model_name=None, model_name_suffix=None):
    if model_name is None:
        model_name = model.__class__.__name__

    model_path = model_name
    if model_dir is not None:
        model_path = model_dir + model_name

    if model_name_suffix is not None:
        model_path += model_name_suffix

    model_path += '.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    logging.info('Model saved: ' + model_path)
    return model_path


# Load model
def load_classification_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    logging.info('Model loaded: ' + model_path)
    return model
