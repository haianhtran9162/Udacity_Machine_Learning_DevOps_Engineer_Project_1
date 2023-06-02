'''
	This script is used to test the churn_library.py script. It tests the following functions:
		- import_data
		- perform_eda
		- encoder_helper
		- perform_feature_engineering
		- train_models

	Author: Tran Hai Anh

	Date: 2023.06.01
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
            Test data import - this example is completed for you to assist with the other test functions
            Parameters:
                            import_data (function): function to import data from a .csv file
            Returns:
                            None
    '''
    # Test import of data
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    # Test that the file has at least one record
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
            Test perform eda function
            Parameters:
                            perform_eda (function): function to perform eda on the data
            Returns:
                            None
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing perform_eda import data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda import data: FAILED")
        raise err

    try:
        perform_eda(df)
        image_output_list = ['churn_distribution.png',
                             'customer_age_distribution.png',
                             'marital_status_distribution.png',
                             'total_transaction_distribution.png',
                             'heatmap.png']
        for file in image_output_list:
            logging.info("Testing perform_eda EDA file %s", file)
            assert os.path.exists('./images/eda/' + file)
        logging.info("Testing perform_eda EDA: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda EDA: images not created %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    '''
            Test encoder helper
            Parameters:
                            encoder_helper (function): function to encode categorical variables
            Returns:
                            None
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing encoder_helper import data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper import data: FAILED")
        raise err

    try:
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']
        response = 'Churn'
        num_cols = df.shape[1]

        df = encoder_helper(df, category_lst, response)

        assert df.shape[1] == num_cols + len(category_lst) + 1

        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: FAILED")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
            Test perform_feature_engineering
            Parameters:
                            perform_feature_engineering (function): function to perform feature engineering
            Returns:
                            None
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info(
            "Testing perform_feature_engineering import data: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing perform_feature_engineering import data: FAILED")
        raise err

    try:
        response = 'Churn'
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, response)

        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]

        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: FAILED")
        raise err


def test_train_models(train_models):
    '''
            Test train_models
            Parameters:
                            train_models (function): function to train models
            Returns:
                            None
    '''
    try:
        df = cls.import_data("./data/bank_data.csv")
        logging.info("Testing train_models import data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models import data: FAILED")
        raise err

    try:
        response = 'Churn'
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df, response)

        train_models(X_train, X_test, y_train, y_test)

        image_output_list = ['roc_curve_lrc.png', 'roc_curve_rfc.png',
                             'logistic_regression_train_metric.png',
                             'random_forest_train_metric.png',
                             'feature_importances.png']
        for file in image_output_list:
            logging.info("Testing train_models model file %s", file)
            assert os.path.exists('./images/results/' + file)

        model_output_list = ['rfc_model.pkl', 'lrc_model.pkl']
        for file in model_output_list:
            logging.info("Testing train_models model file %s", file)
            assert os.path.exists('./models/' + file)
        logging.info("Testing train_models model: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models model: FAILED")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
