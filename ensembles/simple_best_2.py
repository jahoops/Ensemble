# input optionally /ensemble/intermediate/merged_train.csv or /ensemble/intermediate/merged_validate.csv or /ensemble/intermediate/merged_test.csv
# by --train or --validate or --test
# for train and validate, the program will for every record output id, the best model, the estimate, and it's accuracy, best model combined, the estimate, and it's accuracy
# a final print output table will show all models and how they contribute by percentage to the best model on record and the best model combined
# for test, the program will output two csv files, submission_record.csv and submission_combined.csv

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.metrics import mean_absolute_error
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

def start(filename, file):
    # the file looks like this:
    #,id,PremiumAmount,NN1,NN2,NN3,NN4,NN5,NN6,NN7,NN8,NN9,NN10,NN11,NN12,NN13,NN14,NN15,NN16,NN17,NN18,NN19,NN20,Amt_catboost,Amt_knn,Amt_lasso,Amt_lightgbm,Amt_linear,Amt_neuralnet,Amt_randomforest,Amt_ridge,Amt_svr,Amt_xgboost
    #0,404339,2386.0,234761,867908,64408,606476,96848,579952,384253,961179,173578,1136502,566054,75186,1028144,1152365,757286,885075,211156,105477,137516,344960,1118.796,1527.6,1109.721,1138.95,1110.142,1154.968,1949.9,1146.506,854.23,1179.079
    #
    # the NN1 through NN20 columns are the id of the nearest neighbors
    # the Amt_catboost through Amt_xgboost columns are the model predictions
    # the PremiumAmount column is the actual premium amount
    # the id column is the id of the record
    # the file is merged with the model predictions
    # the file is merged with the nearest neighbors data
    #
    # the first dataset for train and validate can be calculated with these records
    # for every record, output id, the best model, the estimate, and it's accuracy
    # also keep a sum of "wins" for each model, and a total sum of records for percentage of wins, if necessary
    # so merely look at the model predictions and the actual premium amount and see which model is the best
    # then calculate the accuracy of the best model
    returnedFile = find_closest_model(file)
    # write the returned file to the intermediate directory
    returnedFile.to_csv(f"ensembles/intermediate/{filename}_record.csv", index=False)

def find_closest_model(file):
    # Extract relevant columns
    actual_col = 'PremiumAmount'
    model_cols = ['Amt_catboost', 'Amt_knn', 'Amt_lasso', 'Amt_lightgbm', 'Amt_linear', 'Amt_neuralnet', 'Amt_randomforest', 'Amt_ridge', 'Amt_svr', 'Amt_xgboost']
    
    # Initialize list to store results
    results = []
    
    # Process each record
    for index, row in file.iterrows():
        actual_value = row[actual_col]
        predictions = row[model_cols]
        
        # Find the model with the closest prediction to the actual value
        closest_model = predictions.idxmin(key=lambda x: abs(predictions[x] - actual_value))
        closest_prediction = predictions[closest_model]
        
        # Calculate accuracy
        closest_model_accuracy = 1 - (abs(closest_prediction - actual_value) / actual_value)
        
        # Store the result
        results.append({
            'id': row['id'],
            'closest_model': closest_model,
            'closest_prediction': closest_prediction,
            'closest_model_accuracy': closest_model_accuracy
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print results
    print("Results for each record:")
    print(results_df)
    
    return results_df

#start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")  # Path to the merged data files
    if len(sys.argv) < 2:
        print("Please provide an argument --train or --validate or --test")
        sys.exit(1)
    if sys.argv[1] == "--train":
        filename = "train"
    elif sys.argv[1] == "--validate":
        filename = "validate"
    elif sys.argv[1] == "--test":
        filename = "test"
    file = pd.read_csv(f"ensembles/intermediate/merged_{filename}.csv")
    start(filename, file)
