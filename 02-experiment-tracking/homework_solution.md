# MLflow Homework Solutions

## Q1. Install MLflow
To get started with MLflow, install the MLflow Python package:
1. Create a separate Python environment (e.g., using conda)
2. Install the package using pip or conda
3. Run `mlflow --version` to verify installation
**mlflow, version 2.22.0**
## Q2. Download and Preprocess Data
- Dataset: Green Taxi Trip Records (Jan, Feb, Mar 2023)
- Use `preprocess_data.py` script to:
    - Load data from `<TAXI_DATA_FOLDER>`
    - Fit DictVectorizer on training set (Jan 2023)
    - Save preprocessed datasets and DictVectorizer

Command:
```bash
python preprocess_data.py --raw_data_path <TAXI_DATA_FOLDER> --dest_path ./output
```
How many files were saved to OUTPUT_FOLDER?
**4**

## Q3. Train Model with Autolog
- Model: RandomForestRegressor (Scikit-Learn)
- Using `train.py` script to:
    - Load preprocessed datasets
    - Train model
    - Calculate RMSE on validation set
- Task: Enable MLflow autologging
- Parameter value for `min_samples_split`: 
**3**
## Q4. Launch Tracking Server
Configuration required:
- Local tracking server
- SQLite backend store
- Artifacts folder: "artifacts"
- Additional parameter needed: 
`default-artifact-root`

## Q5. Hyperparameter Tuning
- Using `hpo.py` for hyperparameter optimization
- Experiment name: "random-forest-hyperopt"
- Log validation RMSE for each run
- Best validation RMSE: **5.355041749098929**

## Q6. Model Registry
- Use `register_model.py` to:
    - Select top 5 runs
    - Calculate test RMSE (March 2023 data)
    - Register best model in model registry
- Test RMSE of best model: **5.355041749098929**
