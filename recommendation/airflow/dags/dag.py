import airflow
import datetime
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator

YESTERDAY = datetime.datetime.now() - datetime.timedelta(days=1)
default_args = {
    'owner': 'Composer Example',
    'depends_on_past': False,
    'email': [''],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'start_date': YESTERDAY,
    'project_id': Variable.get('project_id')
}

with airflow.DAG(
        'asl_ml_pipeline',
        'catchup=False',
        default_args=default_args,
        schedule_interval=datetime.timedelta(days=1)) as dag:

    # Print the dag_run id from the Airflow logs
    print_dag_run_conf = BashOperator(
        task_id='print_dag_run_conf', bash_command='echo {{ dag_run.id }}')

    # TODO: 以下のタスクを加える
    # TODO: DataFlowでの前処理とML Engineへのtraining jobのsubmit、modelのdeploy

    end = DummyOperator(task_id='end')

    print_dag_run_conf >> end
