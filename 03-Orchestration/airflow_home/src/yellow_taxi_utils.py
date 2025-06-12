import pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import mlflow

# Ensure a 'models' folder exists at project_root/models
models_folder = Path(__file__).parent.parent / "models"
models_folder.mkdir(exist_ok=True, parents=True)

def read_dataframe(year: int, month: int):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)
    
    print(f"Loaded {len(df)} records from the dataset")

    # Compute duration in minutes
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    # Filter trips between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    print(f"After filtering, dataset size: {len(df)} records")

    # Cast to string and create composite key
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

def prepare_features(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv

def train_model(X_train, y_train, X_val, y_val, dv):
    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("yellow-taxi-experiment")

    with mlflow.start_run() as run:
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate RMSE
        rmse = ((y_val - y_pred) ** 2).mean() ** 0.5
        print(f"RMSE: {rmse}")

        # Log parameters and metrics
        mlflow.log_params({"model_type": "linear_regression"})
        mlflow.log_metric("rmse", rmse)

        # Save the DictVectorizer
        preprocessor_path = models_folder / "yellow_preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        # Save the model
        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

        return run.info.run_id

def run_training(year: int, month: int) -> str:
    # Read and prepare data
    df = read_dataframe(year=year, month=month)
    
    # Prepare features
    X, dv = prepare_features(df)
    y = df['duration'].values

    # Split data (using first 80% for training)
    n = len(X)
    n_train = int(0.8 * n)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:]
    y_val = y[n_train:]

    # Train model and log with MLflow
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id 