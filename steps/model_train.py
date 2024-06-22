from zenml.steps import BaseStep, step
from sklearn.base import RegressorMixin
import pandas as pd
from model.model_dev import HyperparameterTuner, LightGBMModel, RandomForestModel
import mlflow

class TrainModelStep(BaseStep):
    def entrypoint(self, x_train: pd.DataFrame, y_train: pd.Series, config):
        # Choose the model based on the configuration
        model = LightGBMModel() if config.model_name == "lightgbm" else RandomForestModel()
        mlflow.start_run()
        model.fit(x_train, y_train)
        mlflow.end_run()
        return model

# Example configuration usage, this should be dynamically adjusted based on actual use
train_model_step = TrainModelStep()
