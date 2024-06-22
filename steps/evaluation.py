from zenml.steps import BaseStep, step
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

class EvaluationStep(BaseStep):
    def entrypoint(self, model, x_test: pd.DataFrame, y_test: pd.Series):
        predictions = model.predict(x_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        return r2, mse

# Instantiate the step to use in the pipeline
evaluation_step = EvaluationStep()
