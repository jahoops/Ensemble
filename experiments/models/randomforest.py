from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, train_test_split
from experiments import data_manager, safe_rmsle
from experiments.base_experiment import BaseExperiment  # Updated import
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class RandomForestExperiment(BaseExperiment):
    def __init__(self, output_dir, test_mode=False, sample_size=100000):
        super().__init__(output_dir, test_mode)  # Corrected super() call
        self.sample_size = sample_size

    @property
    def name(self) -> str:
        return "RandomForest"
    
    def create_model(self):
        """Instantiate the Random Forest model."""
        return Pipeline([
            ('random_forest', RandomForestRegressor(random_state=42))
        ])
        
    def get_param_distributions(self):
        if self.test_mode:
            return {
                'random_forest__n_estimators': [50],
                'random_forest__max_depth': [None],
                'random_forest__min_samples_split': [2],
                'random_forest__min_samples_leaf': [1]
            }
        else:
            return {
                'random_forest__n_estimators': [100, 200],
                'random_forest__max_depth': [None, 10, 20],
                'random_forest__min_samples_split': [2, 5, 10],
                'random_forest__min_samples_leaf': [1, 2, 4]
            }

    def transform_data(self, data, dataset_type):
        """Apply model-specific transformations"""
        # Merge distance features
        nn_data = pd.read_csv(f'data/processed/nn_{dataset_type}.csv')
        data = pd.merge(data, nn_data, on='id', how='left')

        # Create distance features
        dist_cols = [f'DIST{i}' for i in range(1, 21)]
        data['avg_dist'] = data[dist_cols].mean(axis=1)
        data['min_dist'] = data[dist_cols].min(axis=1)

        # Keep important features
        keep_cols = ['id', 'HealthScore', 'AnnualIncome', 'CreditScore', 'Age',
                    'avg_dist', 'min_dist'] + dist_cols
        if dataset_type != 'test' and 'PremiumAmount' in data.columns:
            keep_cols.append('PremiumAmount')  # Ensure PremiumAmount is kept
        
        return data[keep_cols]
    
    def run(self):
        """Run the Random Forest experiment and return the trained model."""
        try:
            start_time = time.time()
            self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")
            
            # Load data
            data = data_manager.finaltrain.copy()
            X = self.dropIdPremium(data)
            y = data['PremiumAmount']

            cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
            param_distributions = self.get_param_distributions()
            self.logger.info(f"Param distributions: {param_distributions}")

            search = RandomizedSearchCV(
                estimator=self.create_model(),
                param_distributions=param_distributions,
                n_iter=1 if self.test_mode else 10,
                scoring='neg_mean_squared_error',
                cv=cv,
                verbose=1
            )
            search.fit(X, y)
            trained_model = search.best_estimator_
            self.logger.info(f"Best params: {search.best_params_}")

            # Add predictions for train/validate/test
            self.add_predictions(trained_model, data_manager.finaltrain, 'finaltrain')
            self.add_predictions(trained_model, data_manager.finalvalidate, 'finalvalidate')
            self.add_predictions(trained_model, data_manager.finaltest, 'finaltest')

            end_time = time.time()
            elapsed = end_time - start_time
            self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
            return trained_model
                
        except Exception as e:
            self.logger.error(f"RandomForest run failed: {str(e)}")
            raise

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        predictions = model.predict(X)

        # Standard deviation across RandomForest estimators:
        rf = model.named_steps['random_forest']
        all_estimator_preds = [est.predict(X) for est in rf.estimators_]
        pred_std = np.std(all_estimator_preds, axis=0)

        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
            filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
            self.logger.info(f"{split_name} score: {score:.3f}")
        else:
            filename = f"{self.name}_{split_name}_estimates.csv"

        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3),
            'Confidence': pred_std.round(3)
        })

        output_path = self.output_dir / filename
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Saved {split_name} predictions to {output_path}")

def run_randomforest(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = RandomForestExperiment(output_path, test_mode=test_mode)
    model_info = experiment.run()
    return model_info

if __name__ == "__main__":
    run_randomforest('output', test_mode=True)