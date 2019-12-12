import argparse
import json
import os
from .estimator_with_category import train_and_evaluate


def execute():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--eval_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer_num', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--kernel_regularizer', type=float)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--train_steps', type=int)
    parser.add_argument('--eval_interval_step', type=int)
    parser.add_argument('--category_weight', type=float)

    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )

    params = {k: v for k, v in parser.parse_args().__dict__.items() if v is not None}
    print(params)

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        params['output_dir'],
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('trial', '')
    )
    # Run the training job
    train_and_evaluate(output_dir, params)

    
if __name__ == '__main__':
    execute()
