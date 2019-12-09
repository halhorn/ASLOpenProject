import os
import airflow
import datetime
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.operators.dataflow_operator import DataFlowPythonOperator
from airflow.operators.slack_operator import SlackAPIPostOperator
from airflow import configuration
from airflow.utils.trigger_rule import TriggerRule


YESTERDAY = datetime.datetime.now() - datetime.timedelta(days=1)
DATAFLOW_FILE = os.path.join(
    configuration.get('core', 'dags_folder'), 'dataflow', 'extract.py')

default_args = {
    'owner': 'Composer Example',
    'depends_on_past': False,
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'start_date': YESTERDAY,
    'project_id': Variable.get('project_id'),
    'dataflow_default_options': {
        'project': Variable.get('project_id'),
        'temp_location': 'gs://us-central1-hiroki-kurasawa-772758ea-bucket/temp',
        'runner': 'DataflowRunner'
    }
}

with airflow.DAG(
        'asl_ml_pipeline',
        'catchup=False',
        default_args=default_args,
        schedule_interval=datetime.timedelta(days=1)) as dag:

    # Print the dag_run id from the Airflow logs
    start = DummyOperator(task_id='start')

    post_success_slack = SlackAPIPostOperator(
        task_id='post-success-to-slack',
        token=Variable.get('slack_access_token'),
        text='Hello Airflow',
        channel='#feed'
    )

    post__fail_slack = SlackAPIPostOperator(
        task_id='post-fail-to-slack',
        token=Variable.get('slack_access_token'),
        trigger_rule=TriggerRule.ONE_FAILED,
        text='Hello World!',
        channel='#feed'
    )

    job_args = {
        'output': 'gs://dev-recommend/preprocess'
    }
    data_flow = DataFlowPythonOperator(
        task_id='submit-job-data-flow',
        py_file=DATAFLOW_FILE,
        options=job_args
    )

    end = DummyOperator(task_id='end')

    start >> data_flow >> [post_success_slack, post__fail_slack] >> end
