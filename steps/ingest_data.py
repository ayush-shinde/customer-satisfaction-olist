from zenml.steps import BaseStep, step
import pandas as pd

class IngestDataStep(BaseStep):
    def entrypoint(self) -> pd.DataFrame:
        # Assuming this function correctly reads and returns a DataFrame
        return pd.read_csv("./data/olist_customers_dataset.csv")

# Instantiate the step to use in the pipeline
ingest_data_step = IngestDataStep()
