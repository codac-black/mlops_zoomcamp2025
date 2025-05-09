# NYC Taxi Trip Duration Prediction Project

## Project Overview
This project implements a machine learning model to predict taxi trip durations in New York City using the NYC Yellow Taxi Trip Records dataset. The model estimates how long a taxi journey will take based on pickup and dropoff locations.

## Technical Implementation

### Data Processing
- **Dataset**: NYC Yellow Taxi Trip Records from January and February 2023.
- **Steps**:
    - Computed trip durations from pickup and dropoff timestamps.
    - Removed outliers by filtering trips between 1-60 minutes.
    - Converted location IDs to categorical features using one-hot encoding.

### Model Development

#### Feature Engineering
- Transformed pickup and dropoff location IDs using `DictVectorizer`.
- Created sparse matrix representation for efficient processing.

#### Model Selection
- Implemented Linear Regression as a baseline model.
- Used scikit-learn's implementation for reproducibility.

### Evaluation Metrics
- Evaluated model performance using Root Mean Square Error (RMSE).
- Validated results on both training (January) and validation (February) datasets.

## Key Learnings
- Demonstrated effective data preprocessing techniques for time-series data.
- Implemented feature engineering for categorical variables.
- Applied cross-validation techniques using temporal split.
- Gained practical experience with scikit-learn's ML pipeline.

## Technologies Used
- **Python**:
    - `pandas` for data manipulation.
    - `scikit-learn` for machine learning.
    - Jupyter Notebooks for development.

## Future Improvements
- Implement more advanced models (e.g., Random Forests, XGBoost).
- Add feature engineering for temporal patterns.
- Deploy the model as a web service.
- Implement real-time prediction capabilities.

This project is part of the **MLOps Zoomcamp 2025** coursework, focusing on practical machine learning implementation and operations.

## Link to GitHub Repository
- [GitHub Repository](https://github.com/codac-black/mlops_zoomcamp2025)