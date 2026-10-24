import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.metrics import mean_absolute_error
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

def getModelFileList(model_names, type_of_data):
    models_data = {}
    for model_name in model_names:
        file = f"ensembles/inputs/{model_name}/{type_of_data}.csv"
        if os.path.exists(file):
            models_data[model_name] = pd.read_csv(file)
    return models_data

def getTrainValidateTestSets(data_dir, model_names):
    train = data_manager.finaltrain.copy();
    validate = data_manager.finalvalidate.copy();
    test = data_manager.finaltest.copy();
    nn_train = data_manager.nn_train.copy();
    nn_validate = data_manager.nn_validate.copy();
    nn_test = data_manager.nn_test.copy();
    # only take the id and premium amount columns from train, validate, and test
    train = train[['id', 'PremiumAmount']]
    validate = validate[['id', 'PremiumAmount']]
    test = test[['id', 'PremiumAmount']]
    # merge the train, validate, and test data with only the NN1 through NN20 columns of the nearest neighbors data
    train = train.merge(nn_train[['id', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'NN6', 'NN7', 'NN8', 'NN9', 'NN10', 'NN11', 'NN12', 'NN13', 'NN14', 'NN15', 'NN16', 'NN17', 'NN18', 'NN19', 'NN20']], on='id')
    validate = validate.merge(nn_validate[['id', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'NN6', 'NN7', 'NN8', 'NN9', 'NN10', 'NN11', 'NN12', 'NN13', 'NN14', 'NN15', 'NN16', 'NN17', 'NN18', 'NN19', 'NN20']], on='id')
    test = test.merge(nn_test[['id', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5', 'NN6', 'NN7', 'NN8', 'NN9', 'NN10', 'NN11', 'NN12', 'NN13', 'NN14', 'NN15', 'NN16', 'NN17', 'NN18', 'NN19', 'NN20']], on='id')
    # merge the train, validate, and test data with the model predictions
    file_paths = [
        {'file': train, 'name': 'train'},
        {'file': validate, 'name': 'validate'},
        {'file': test, 'name': 'test'}
    ]
    for file in file_paths:
        model_files = getModelFileList(model_names, file['name'])
        for model_name in model_files:
            # rename Premium Amount column to Amt_{model_name}
            # only id and Premium Amount columns are taken from the model file
            model_files[f"{model_name}"] = model_files[f"{model_name}"][['id', 'Premium Amount']]
            # where model_name is the name of the model
            model_files[f"{model_name}"] = model_files[f"{model_name}"].rename(columns={'Premium Amount': f'Amt_{model_name}'})
            # for example, if model_name is 'catboost', the column will be 'Amt_catboost'
            file['file'] = file['file'].merge(model_files[f"{model_name}"], on='id')
        # save these three files to the ensembles/intermediate directory as csv
        file['file'].to_csv(f"{data_dir}/merged_{file['name']}.csv")

#start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")  # Path to the merged data files
    model_names = ['catboost', 'knn', 'lasso', 'lightgbm', 'linear', 'neuralnet', 'randomforest', 'ridge', 'svr', 'xgboost']
    getTrainValidateTestSets(data_dir, model_names)
    print("Data sets created")





