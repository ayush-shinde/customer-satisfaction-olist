from zenml.steps import BaseStep, step
import pandas as pd

class CleanDataStep(BaseStep):
    def entrypoint(self, data: pd.DataFrame):
        # Assuming DataCleaning performs cleaning and returns train-test split
        from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        return data_cleaning.handle_data()

# Instantiate the step to use in the pipeline
clean_data_step = CleanDataStep()
