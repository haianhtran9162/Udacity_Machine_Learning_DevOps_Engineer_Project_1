'''
    This script contains functions for the churn library

    Author: Tran Hai Anh

    Date: 2023.06.01
'''

# import libraries
import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
        Returns dataframe for the csv found at pth
        Parameters:
                pth: a path to the csv
        Returns:
                df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(dataframe):
    '''
        Perform eda on df and save figures to images folder
        Parameters:
                df_origin: pandas dataframe

        Returns:
                None
    '''
    df = dataframe.copy()

    # Add churn column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # EDA Churn column and save figure
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.png')

    # EDA Customer_Age and save figure
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    # EDA Marital_Status and save figure
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    # EDA Total_Trans_Ct and save figure
    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(fname='./images/eda/total_transaction_distribution.png')

    # Headmap of correlation matrix and save figure
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')

    return None


def encoder_helper(dataframe, category_lst, response):
    '''
        Helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        Parameters:
                dataframe: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name

        Returns:
                df: pandas dataframe with new columns for
    '''
    df = dataframe.copy()
    
    # Add Churn column
    df[response] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    for col in category_lst:
        col_lst = []
        col_groups = df.groupby(col).mean()[response]
        for val in df[col]:
            col_lst.append(col_groups.loc[val])
        if response:
            df[col + '_' + response] = col_lst
        else:
            df[col] = col_lst
    return df


def perform_feature_engineering(dataframe, response):
    '''
        Feature engineering dataframe and split into train and test sets
        Parameters:
                df: pandas dataframe
                response: string of response name

        Returns:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
    '''
    # List of columns that contain categorical features
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # Feature engineering categorical columns
    df = encoder_helper(dataframe, cat_columns, response)

    y = df['Churn']
    X = pd.DataFrame()

    # List of columns to keep
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
        Produces classification report for training and testing results and stores report as image
        in images folder
        Parameters:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        Returns:
                None
    '''
    # Logistic Regression Classification Report and save figure
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_regression_train_metric.png')

    # Random Forest Classification Report and save figure
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(fname='./images/results/random_forest_train_metric.png')

    return None


def feature_importance_plot(model, X_data, output_pth):
    '''
        Creates and stores the feature importances in pth
        Parameters:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        Returns:
                None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')

    return None


def train_models(X_train, X_test, y_train, y_test):
    '''
        Train, store model results: images + scores, and store models
        Parameters:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        Returns:
                None
    '''
    # Define ML models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Definf search space for ML models
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Grid Search and fit for RandomForestClassifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Fit to LogisticRegression
    lrc.fit(X_train, y_train)

    # Apply best estimator and compute predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Compute predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/lrc_model.pkl')

    # ROC Curve and save figure
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    lrc_plot.plot(ax=ax, alpha=0.8)
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve.png')

    # Classification Report and save figure
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Feature Importance Plot and save figure
    feature_importance_plot(cv_rfc, X_test, './images/results/')


if __name__ == '__main__':
    # Import data
    df = import_data('./data/bank_data.csv')

    # Perform EDA
    perform_eda(df)

    # Perform Feature Engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    # Train models
    train_models(X_train, X_test, y_train, y_test)
