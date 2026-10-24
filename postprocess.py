import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from utils import feature_cols, data_manager  # Ensure correct import
import logging

def postprocess():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    base_path = "data/processed"

    cleaned_train = data_manager.loadtrain.copy()
    cleaned_validate = data_manager.loadvalidate.copy()
    cleaned_test = data_manager.loadtest.copy()

    feature_files = {
        "train": "feature_train.csv",
        "validate": "feature_validate.csv",
        "test": "feature_test.csv"
    }
    nn_files = {
        "train": "nn_train.csv",
        "validate": "nn_validate.csv",
        "test": "nn_test.csv"
    }
    final_files = {
        "train": "final_train.csv",
        "validate": "final_validate.csv",
        "test": "final_test.csv"
    }
    # Include required columns
    required_columns = set(["id", "PremiumAmount"])
    keep_cols = set(feature_cols).union(required_columns)

    logger.info("Starting post-processing...")

    # Iterate through the datasets and process them
    for split, df in [('train', cleaned_train), ('validate', cleaned_validate), ('test', cleaned_test)]:
        if df is None:
            logger.error(f"{split.upper()} dataset is empty. Skipping post-processing.")
            raise ValueError(f"{split.upper()} dataset is empty.")
        
        keep_split = [c for c in df.columns if c in keep_cols]
        df = df[keep_split]

        missing_cols_split = keep_cols - set(keep_split)
        if missing_cols_split:
            logger.warning(f"{split.upper()} is missing columns: {missing_cols_split}. Adding them with default values.")
            for col in missing_cols_split:
                if col == "PremiumAmount" and split == "test":
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(0)

        df = df.copy()

        # Verify columns
        if sorted(df.columns) != sorted(keep_cols):
            logger.error(f"{split.upper()} columns mismatch. Expected: {sorted(keep_cols)}, Got: {sorted(df.columns)}")
            raise ValueError(f"Columns mismatch in {split.upper()} dataset.")

        # Save processed features
        output_path = os.path.join(base_path, feature_files[split])
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed features to {output_path}")

    # all nearest neighbors are from data_manager.train, use the id from data_manager.train
    # find the nearest neighbors for each row in the train, validate, and test datasets
    # save the nearest neighbors to a nn_files dictionary train, validate, and test
    # use nn_id and nn_dist as the column names for the nearest neighbor id and distance

    n_neighbors = 20  # As per your requirement
    logger.info(f"Initializing NearestNeighbors with training data for {n_neighbors} neighbors.")
    X_train_nn = data_manager.loadtrain.select_dtypes(include=[np.number]).drop(columns=["id", "PremiumAmount"], errors='ignore')
    sorted_columns = sorted(X_train_nn.columns)
    X_train_nn = X_train_nn[sorted_columns]
    nn = NearestNeighbors(n_neighbors=n_neighbors+1, metric="euclidean")  # +1 to exclude self
    nn.fit(X_train_nn)

    # Calculate nearest neighbors for Training Set
    logger.info("Calculating nearest neighbors for TRAIN set...")
    # For training set, neighbors are within the training data
    distances_train, indices_train = nn.kneighbors(X_train_nn)
    nn_df_train = pd.DataFrame(index=data_manager.loadtrain.index)

    for i in range(1, n_neighbors+1):  # Skip first neighbor (self)
        logger.info(f"Calculating NN{i} for TRAIN set...")
        # Get indices of nearest neighbors
        nn_indices = indices_train[:, i]
        
        # Get values for neighbor i
        train_nn_ids = data_manager.loadtrain.iloc[nn_indices]["id"].values
        train_nn_distances = distances_train[:, i]
        train_nn_premiums = data_manager.loadtrain.iloc[nn_indices]["PremiumAmount"].values
        
        # Store in dataframe
        nn_df_train[f"NN{i}"] = train_nn_ids
        nn_df_train[f"DIST{i}"] = train_nn_distances
        nn_df_train[f"PRED{i}"] = train_nn_premiums

    # Combine with original ids
    nn_output_path_train = os.path.join(base_path, nn_files["train"])
    nn_final_df_train = pd.concat([
        data_manager.loadtrain[["id"]].reset_index(drop=True),
        nn_df_train.reset_index(drop=True)
    ], axis=1)
    nn_final_df_train.to_csv(nn_output_path_train, index=False)
    logger.info(f"Saved nearest neighbors to {nn_output_path_train}")

    def calculate_nn(ff, split_name, sorted_columns):
        logger.info(f"Processing nearest neighbors for {split_name.upper()}...")

        if len(ff) > n_neighbors and len(X_train_nn) > 0:
            logger.info(f"Calculating nearest neighbors for {split_name.upper()} with {n_neighbors} neighbors...")
            X_nn = ff[sorted_columns]  # Select columns in same order as training set
            distances, indices = nn.kneighbors(X_nn)

            # Create nearest neighbors dataframe with distances and scores
            nn_df = pd.DataFrame(index=ff.index)

            for i in range(1, n_neighbors+1):
                logger.info(f"Calculating NN{i} for {split_name.upper()}...")
                nn_indices = indices[:, i-1]
                nn_ids = data_manager.loadtrain.iloc[nn_indices]["id"].values
                nn_distances = distances[:, i-1]
                nn_scores = data_manager.loadtrain.iloc[nn_indices]["PremiumAmount"].values
                
                nn_df[f"NN{i}"] = nn_ids
                nn_df[f"DIST{i}"] = nn_distances
                nn_df[f"PRED{i}"] = nn_scores

            # Combine with original ids
            nn_output_path = os.path.join(base_path, nn_files[split_name])
            nn_final_df = pd.concat([
                ff[["id"]].reset_index(drop=True),
                nn_df.reset_index(drop=True)
            ], axis=1)
            nn_final_df.to_csv(nn_output_path, index=False)
            logger.info(f"Saved nearest neighbors to {nn_output_path}")
        else:
            logger.warning(f"Skipping nearest neighbors for {split_name.upper()}: not enough rows or no numeric columns.")


    calculate_nn(data_manager.loadvalidate, 'validate', sorted_columns)
    calculate_nn(data_manager.loadtest, 'test', sorted_columns)   

if __name__ == "__main__":
    postprocess()
