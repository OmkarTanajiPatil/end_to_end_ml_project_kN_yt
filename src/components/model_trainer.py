
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor


import os
import sys
import pandas as pd

from sklearn.metrics import r2_score

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                pd.DataFrame(train_array[:, :-1]),
                train_array[:, -1],
                pd.DataFrame(test_array[:, :-1]),
                test_array[:, -1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(
                    verbose=False, allow_writing_files=False
                ),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "LGBM Regressor": LGBMRegressor(verbosity=-1),
                "KNeighbors Regressor": KNeighborsRegressor(),
            }
            
            params = {
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.1, 1.0, 10.0]},
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_depth': [3, 5, 10, None]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_depth': [3, 5, 10, None]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'subsample': [0.6, 0.8, 1.0]
                },
                "CatBoosting Regressor": {
                    'depth': [4, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'iterations': [100, 200, 300]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200]
                },
                "LGBM Regressor": {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 200, 300],
                    'num_leaves': [20, 31, 50]
                },
                "KNeighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
            
            model_report : dict = evaluate_models(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                models = models,
                params = params
            )
            
            logging.info(f"Models being trained and Report is generated.")
            
            #$ Get the best model score from the dict
            best_model_score = max(sorted(model_report.values()))
            
            #$ Get the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found", sys)

            logging.info(f"Training completed -> Best Model: {best_model} Score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2 = r2_score(y_test, predicted)
            return r2
                        
        except Exception as e:
            raise CustomException(e, sys)











