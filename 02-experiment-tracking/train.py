import os
import pickle
import click
import mlflow
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_pickle(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found")
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    try:
        logger.info("Starting the training pipeline...")
        
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("nyc-taxi-experiment")
        mlflow.sklearn.autolog()

        logger.info("Loading datasets...")
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")

        logger.info("Training random forest model...")
        with mlflow.start_run() as run:
            logger.info(f"MLflow run ID: {run.info.run_id}")
            
            rf = RandomForestRegressor(max_depth=10, random_state=0, n_jobs=1)
            rf.fit(X_train, y_train)
            
            logger.info("Making predictions...")
            y_pred = rf.predict(X_val)

            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            logger.info(f"RMSE: {rmse}")
            
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    run_train()