from pipelines.training_pipeline import TrainingPipeline
from config import ModelNameConfig

if __name__ == "__main__":
    config = ModelNameConfig(model_name="randomforest", fine_tuning=True)
    training_pipeline = TrainingPipeline(
        ingest_data=ingest_data_step(),
        clean_data=clean_data_step(),
        train_model=train_model_step(config=config),
        evaluation=evaluation_step()
    )

    result = training_pipeline.run()
    print(result)
    print(
        "Now run \n"
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        " experiment. Here you'll also be able to compare the two runs."
    )
