import time
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, plot_roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn import svm

from src.common.autoenc_feature_extraction_util import encode_data_with_autoencoder
from src.common.model_util import save_classification_model


def create_model(model, x_train, y_train):
    model_name = model.__class__.__name__
    start_time = time.time()

    logging.info("---> Running " + model_name + " . . .")

    model.fit(x_train, y_train)

    exec_time_seconds = (time.time() - start_time)
    exec_time_minutes = exec_time_seconds / 60

    print("---> END in %s seconds = %s minutes ---" % (exec_time_seconds, exec_time_minutes))
    return model


def print_score(y_test, predictions):
    print("---> ---> Accuracy Score: ", accuracy_score(predictions, y_test))


def print_class_report(y_test, predictions):
    cls_report = classification_report(y_test, predictions);
    print("---> ---> Classification report: \n", cls_report)
    # plot_classification_report(cls_report)


def cross_val(x_train, y_train):
    print("---> ---> Running cross validation . . . ")
    clf = svm.SVC(kernel='linear', C=1, random_state=42)
    scores = cross_val_score(clf, x_train, y_train, cv=10)
    print(scores)


def print_stats(model, X_test, y_test):
    predictions = model.predict(X_test)
    print_score(y_test, predictions)
    print_class_report(y_test, predictions)
    plot_confusion_matrix(model, X_test, y_test)


def run_classification_algorithms(x_train, y_train, x_test, y_test, model_output_dir=None, model_name_suffix='_std'):
    # https://towardsdatascience.com/a-quick-overview-of-5-scikit-learn-classification-algorithms-33fdc11ab0b9
    # https://scikit-learn.org/stable/modules/cross_validation.html
    # cross_val(x_train, y_train)

    # # LogisticRegression
    lr_model = create_model(LogisticRegression(solver='lbfgs', max_iter=1000), x_train, y_train)  # 2 min
    print_stats(lr_model, x_test, y_test)
    save_classification_model(lr_model, model_output_dir, model_name_suffix=model_name_suffix)

    # Perceptron
    perceptron = Perceptron(eta0=0.2, max_iter=1000, tol=1e-3, verbose=0, early_stopping=True,
                            validation_fraction=0.1)
    prc_model = create_model(perceptron, x_train, y_train)  # 15 sec
    print_stats(prc_model, x_test, y_test)
    save_classification_model(prc_model, model_output_dir, model_name_suffix=model_name_suffix)

    # GaussianNB
    gnb_model = create_model(GaussianNB(), x_train, y_train)  # 1.5 sec
    print_stats(prc_model, x_test, y_test)
    save_classification_model(prc_model, model_output_dir, model_name_suffix=model_name_suffix)

    # DecisionTreeClassifier
    dec_tree_model = create_model(DecisionTreeClassifier(random_state=0), x_train, y_train)  # 5 sec
    print_stats(dec_tree_model, x_test, y_test)
    save_classification_model(dec_tree_model, model_output_dir, model_name_suffix=model_name_suffix)

    # RandomForestClassifier
    rd_forest_model = create_model(RandomForestClassifier(random_state=0), x_train, y_train)  # 3 min
    print_stats(rd_forest_model, x_test, y_test)
    save_classification_model(rd_forest_model, model_output_dir, model_name_suffix=model_name_suffix)

    # GradientBoostingClassifier
    grd_boost_model = create_model(GradientBoostingClassifier(random_state=0), x_train, y_train)  # 10 min
    print_stats(grd_boost_model, x_test, y_test)
    save_classification_model(grd_boost_model, model_output_dir, model_name_suffix=model_name_suffix)

    # KNeighborsClassifier
    kneighbors_model = create_model(KNeighborsClassifier(n_neighbors=3), x_train, y_train)  # 55 min
    print_stats(kneighbors_model, x_test, y_test)
    save_classification_model(kneighbors_model, model_output_dir, model_name_suffix=model_name_suffix)

    # DecisionTreeRegressor
    # dec_tree_reg_model = create_model(DecisionTreeRegressor(), x_train, y_train)  # exc!
    # print_stats(dec_tree_reg_model, x_test, y_test)
    # save_classification_model(dec_tree_reg_model, model_output_dir, model_name_suffix=model_name_suffix)
    # # Support Vector Classification
    # model = svm.SVC(kernel='linear', C=1, random_state=0)
    # run_alg(model, x_train, x_test, y_train, y_test)
    # print(" === End Training === ")


def create_classification_models(X_train, y_train, x_test, y_test, model_output_dir=None, model_name_suffix='_std'):
    print('------ Running classification... ----')

    # Run classification algorithms
    run_classification_algorithms(X_train, y_train, x_test, y_test, model_output_dir=model_output_dir, model_name_suffix=model_name_suffix)


def create_classification_models_with_ae_features(X_train, y_train, x_test, y_test, encoder_path, model_output_dir=None, model_name_suffix='_ae'):
    print('------ Running classification with autoencoded features... ----')

    # Encode X with autoencoder
    X_train_encode, y_train, X_test_encode, y_test = encode_data_with_autoencoder(X_train, y_train, x_test, y_test, encoder_path=encoder_path)

    # Run classification algorithms
    run_classification_algorithms(X_train_encode, y_train, X_test_encode, y_test, model_output_dir=model_output_dir, model_name_suffix=model_name_suffix)
