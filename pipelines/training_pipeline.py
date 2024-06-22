from zenml.pipelines import BasePipeline
from steps.ingest_data import ingest_data_step
from steps.clean_data import clean_data_step
from steps.model_train import train_model_step
from steps.evaluation import evaluation_step

class TrainingPipeline(BasePipeline):
    ingest_data = ingest_data_step
    clean_data = clean_data_step
    train_model = train_model_step
    evaluation = evaluation_step

    def connect(self):
        df = self.ingest_data()
        x_train, x_test, y_train, y_test = self.clean_data(data=df)
        model = self.train_model(x_train=x_train, y_train=y_train)
        r2_score, mse = self.evaluation(model=model, x_test=x_test, y_test=y_test)
        return r2_score, mse
