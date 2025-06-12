import sys
from pathlib import Path
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum

# Add parent directory of 'src' to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 1) Define default_args
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# 2) Instantiate the DAG
with DAG(
    dag_id="nyc_taxi_monthly_training",
    default_args=default_args,
    description="Monthly pipeline: read, prep, train XGBoost, register with MLflow",
    schedule_interval="@monthly",               # shorthand for "0 0 1 * *"
    start_date=pendulum.datetime(2023, 3, 1, tz="UTC"),
    catchup=True,                               # enable backfill for past months
    max_active_runs=1,                          # avoid multiple concurrent runs
    tags=["nyc", "taxi", "ml", "xgboost", "mlflow"],
) as dag:

    def _train_for_execution_date(execution_date, **context):
        """
        This callable is what PythonOperator will run.
        We extract year/month from the execution_date and call run_training().
        """
        # Import here to avoid import during DAG parsing
        from src.nyc_taxi_util import run_training

        # execution_date is a pendulum.DateTime in UTC
        year = execution_date.year
        month = execution_date.month

        # Call the utility function
        run_id = run_training(year=year, month=month)

        # Push run_id to XCom so you can see it in the UI
        context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
        return run_id

    # 3) PythonOperator that runs the pipeline
    train_task = PythonOperator(
        task_id="run_monthly_training",
        python_callable=_train_for_execution_date,
        provide_context=True,
    )

    # If you had more steps (e.g. "evaluate_task", "deploy_task"), you could chain them:
    # train_task >> evaluate_task >> deploy_task

    # Here, we only have a single-step pipeline, so no downstream tasks.
    train_task
