import pickle
from pathlib import Path

# Ensure a 'models' folder exists at project_root/models
models_folder = Path(__file__).parent.parent / "models"
models_folder.mkdir(exist_ok=True, parents=True)

def read_dataframe(year: int, month: int):
    # Import here to avoid import during module load
    import pandas as pd
    
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df = pd.read_parquet(url)

    # Compute duration in minutes
    df["duration"] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]

    # Cast to string and create composite key
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


def create_X(df, dv=None):
    # Import here to avoid import during module load
    from sklearn.feature_extraction import DictVectorizer
    
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]
    dicts = df[categorical + numerical].to_dict(orient="records")

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def train_xgboost_model(X_train, y_train, X_val, y_val, dv):
    # Import here to avoid import during module load
    import xgboost as xgb
    from sklearn.metrics import root_mean_squared_error
    import mlflow

    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    with mlflow.start_run() as run:
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        valid_dmatrix = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            "learning_rate": 0.09585355369315604,
            "max_depth": 30,
            "min_child_weight": 1.060597050922164,
            "objective": "reg:linear",
            "reg_alpha": 0.018060244040060163,
            "reg_lambda": 0.011658731377413597,
            "seed": 42,
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train_dmatrix,
            num_boost_round=30,
            evals=[(valid_dmatrix, "validation")],
            early_stopping_rounds=50,
        )

        y_pred = booster.predict(valid_dmatrix)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Save the DictVectorizer
        preprocessor_path = models_folder / "preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

        # Save the XGBoost model
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        return run.info.run_id


def run_training(year: int, month: int) -> str:
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1

    df_train = read_dataframe(year=year, month=month)
    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train, dv=None)
    X_val, _ = create_X(df_val, dv=dv)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    run_id = train_xgboost_model(X_train, y_train, X_val, y_val, dv)
    print(f"MLflow run_id: {run_id}")
    return run_id



